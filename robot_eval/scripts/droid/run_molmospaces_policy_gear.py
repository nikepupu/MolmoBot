"""Run a MolmoSpaces/MolmoBot policy on DROID hardware through gear_droid.

This client is intentionally separate from the policy server:

1. Start the MolmoBot server from ``MolmoBot/`` with ``serve_molmo.py``.
2. Run this script from a DROID/gear_droid robot-control environment.

The script reuses gear_droid's RobotEnv, rollout controls, and video session
recording, while speaking the MolmoBot websocket protocol and observation
schema expected by ``olmo.eval.configure_real_robot.RealRobotVLAPolicy``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


DROID_CONTROL_FREQUENCY = 15
MAX_MLSPACES_GRIPPER_POS = 0.824033
DEFAULT_GEAR_DROID_ROOT = Path("/home/gear/Projects/gr00t/external_dependencies/gear_droid")
_CV2 = None

CAMERA_IDS_BY_SYSTEM = {
    "1": {
        "left": "36308983",
        "right": "34319716",
        "wrist": "17354366",
    },
    "2": {
        "left": "30539522",
        "right": "36991098",
        "wrist": "10764169",
    },
    "ut_austin": {
        "left": "34476795",
        "right": "25047636",
        "wrist": "18659563",
    },
    "gatech": {
        "left": "21497414",
        "right": "20036094",
        "wrist": "18482824",
    },
}


def _default_camera_id(camera: str) -> str | None:
    droid_system = os.environ.get("GEAR_DROID_SYSTEM")
    if droid_system is None:
        return None
    return CAMERA_IDS_BY_SYSTEM.get(droid_system, {}).get(camera)


def _install_gear_droid_paths(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"gear_droid root does not exist: {root}. "
            "Pass --gear-droid-root or set GEAR_DROID_ROOT."
        )

    scripts = root / "scripts"
    for path in (root, scripts):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_cv2():
    global _CV2
    if _CV2 is None:
        import cv2

        _CV2 = cv2
    return _CV2


def _check_camera_runtime(args: argparse.Namespace) -> None:
    try:
        cv2 = _get_cv2()
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV could not be imported in the gear_droid environment. If you are using uv, repair it with:\n"
            "  cd /home/gear/Projects/gr00t/external_dependencies/gear_droid\n"
            "  uv pip uninstall -y opencv-python opencv-contrib-python\n"
            "  uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' msgpack-numpy websockets"
        ) from exc

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV imported, but cv2.aruco is missing. gear_droid needs opencv-contrib-python:\n"
            "  cd /home/gear/Projects/gr00t/external_dependencies/gear_droid\n"
            "  uv pip uninstall -y opencv-python opencv-contrib-python\n"
            "  uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' msgpack-numpy websockets"
        )

    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError(
            "The ZED SDK Python API (pyzed.sl) is not installed in this Python environment. "
            "Action replay can run without it, but hardware inference needs live ZED images. Install it with:\n"
            "  cd /home/gear/Projects/gr00t/external_dependencies/gear_droid\n"
            "  uv run --no-sync python /usr/local/zed/get_python_api.py --path /tmp/zed_py\n"
            "  uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' --reinstall"
        ) from exc

    devices = sl.Camera.get_device_list()
    serials = [str(device.serial_number) for device in devices]
    print(f"Detected ZED camera serials: {serials}")

    if "0" in serials:
        raise RuntimeError(
            "The ZED SDK detected a camera with serial number 0. This usually means a ZED camera "
            "is not enumerating correctly or failed to start its USB stream. For GEAR_DROID_SYSTEM=1 "
            f"we expect wrist={args.wrist_camera_id}, left={args.left_camera_id}, right={args.right_camera_id}. "
            "Unplug/replug the wrist ZED Mini, check that no other process is using the camera, and rerun "
            "the ZED device-list check before starting inference."
        )

    expected = {
        "left": args.left_camera_id,
        "right": args.right_camera_id,
        "wrist": args.wrist_camera_id,
    }
    missing = [f"{name}={serial}" for name, serial in expected.items() if serial not in serials]
    if missing:
        raise RuntimeError(
            "Missing expected ZED camera(s): "
            + ", ".join(missing)
            + f". Detected serials: {serials}. Fix the camera connection or pass explicit camera ids."
        )


def _configure_zed_params(args: argparse.Namespace) -> None:
    if not args.disable_zed_depth:
        return

    import pyzed.sl as sl
    from droid.camera_utils.camera_readers import zed_camera

    zed_camera.standard_params["depth_mode"] = sl.DEPTH_MODE.NONE
    zed_camera.advanced_params["depth_mode"] = sl.DEPTH_MODE.NONE
    print("Configured ZED cameras for RGB-only inference (depth_mode=NONE).")


def _require_camera_ids(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("left_camera_id", "right_camera_id", "wrist_camera_id")
        if getattr(args, name) is None
    ]
    if not missing:
        return

    droid_system = os.environ.get("GEAR_DROID_SYSTEM")
    raise ValueError(
        "Missing camera ids: "
        + ", ".join(missing)
        + ". Set GEAR_DROID_SYSTEM to one of "
        + ", ".join(CAMERA_IDS_BY_SYSTEM)
        + f" or pass explicit --left-camera-id/--right-camera-id/--wrist-camera-id. "
        f"Current GEAR_DROID_SYSTEM={droid_system!r}."
    )


def _extract_observation(
    args: argparse.Namespace,
    obs_dict: dict[str, Any],
    *,
    save_to_disk: bool = False,
) -> dict[str, np.ndarray]:
    """Extract RGB images and proprio state from gear_droid's raw observation."""
    image_observations = obs_dict["image"]
    left_image = right_image = wrist_image = None

    for key, image in image_observations.items():
        if args.left_camera_id in key and args.stereo_side in key:
            left_image = image
        elif args.right_camera_id in key and args.stereo_side in key:
            right_image = image
        elif args.wrist_camera_id in key and args.stereo_side in key:
            wrist_image = image

    if left_image is None or right_image is None or wrist_image is None:
        available = list(image_observations.keys())
        missing = []
        if left_image is None:
            missing.append(f"left camera {args.left_camera_id}")
        if right_image is None:
            missing.append(f"right camera {args.right_camera_id}")
        if wrist_image is None:
            missing.append(f"wrist camera {args.wrist_camera_id}")
        raise ValueError(f"Missing {', '.join(missing)} in observations. Available: {available}")

    # gear_droid camera frames are BGR(A); MolmoBot expects RGB uint8.
    left_image = left_image[..., :3][..., ::-1]
    right_image = right_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    curr_obs = {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.asarray(robot_state["cartesian_position"], dtype=np.float32),
        "joint_position": np.asarray(robot_state["joint_positions"], dtype=np.float32),
        "gripper_position": np.asarray([robot_state["gripper_position"]], dtype=np.float32),
    }

    if save_to_disk:
        cv2 = _get_cv2()
        combined = np.concatenate([left_image, wrist_image, right_image], axis=1)
        cv2.imwrite("robot_camera_views.png", combined[..., ::-1])

    return curr_obs


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    cv2 = _get_cv2()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _molmospaces_gripper_qpos(curr_obs: dict[str, np.ndarray]) -> np.ndarray:
    # gear_droid exposes 0=open, 1=closed. MolmoSpaces qpos uses the same
    # direction, scaled to the training gripper joint range.
    gripper_closed = float(np.clip(curr_obs["gripper_position"][0], 0.0, 1.0))
    return np.full(2, gripper_closed * MAX_MLSPACES_GRIPPER_POS, dtype=np.float32)


def _build_molmospaces_request(
    args: argparse.Namespace,
    curr_obs: dict[str, np.ndarray],
    instruction: str,
) -> dict[str, Any]:
    exo_key = f"{args.external_camera}_image"
    exo_image = _resize_rgb(curr_obs[exo_key], args.input_height, args.input_width)
    wrist_image = _resize_rgb(curr_obs["wrist_image"], args.input_height, args.input_width)

    return {
        "task": instruction,
        "qpos": {
            "arm": curr_obs["joint_position"].astype(np.float32),
            "gripper": _molmospaces_gripper_qpos(curr_obs),
        },
        "exo_camera_1": exo_image,
        "wrist_camera": wrist_image,
    }


def _action_to_robot_command(
    args: argparse.Namespace,
    curr_obs: dict[str, np.ndarray],
    action_dict: dict[str, Any],
) -> np.ndarray:
    arm = np.asarray(action_dict["arm"], dtype=np.float32).reshape(-1)[:7]
    current_arm = curr_obs["joint_position"].astype(np.float32)

    if args.action_scale != 1.0:
        arm = current_arm + (arm - current_arm) * args.action_scale

    if args.max_joint_delta is not None:
        delta = np.clip(arm - current_arm, -args.max_joint_delta, args.max_joint_delta)
        arm = current_arm + delta

    gripper_raw = float(np.asarray(action_dict["gripper"]).reshape(-1)[0])
    gripper = gripper_raw / 255.0 if gripper_raw > 1.0 else gripper_raw
    gripper = float(np.clip(gripper, 0.0, 1.0))
    if args.binarize_gripper:
        gripper = 1.0 if gripper > args.gripper_threshold else 0.0

    return np.concatenate([arm, np.asarray([gripper], dtype=np.float32)])


class MolmoSpacesPolicyClient:
    """Small synchronous client for ``olmo.eval.websocket_server``."""

    def __init__(self, host: str, port: int, *, timeout: float | None = None):
        try:
            import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "MolmoSpacesPolicyClient needs msgpack-numpy and websockets. "
                "Install them in the robot-control environment with: "
                "pip install msgpack-numpy websockets"
            ) from exc

        self.uri = f"ws://{host}:{port}"
        self._connect_fn = connect
        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._timeout = timeout
        self._websocket = None
        self.metadata: dict[str, Any] = {}
        self.connect()

    def connect(self) -> None:
        self.close()
        kwargs = {"compression": None, "max_size": None}
        if self._timeout is not None:
            kwargs["open_timeout"] = self._timeout
        self._websocket = self._connect_fn(self.uri, **kwargs)
        self.metadata = self._unpack(self._websocket.recv())

    def reset(self) -> None:
        self.connect()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._websocket is None:
            self.connect()
        self._websocket.send(self._packer.pack(observation))
        response = self._websocket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned an error:\n{response}")
        return self._unpack(response)

    def close(self) -> None:
        if self._websocket is not None:
            with suppress(Exception):
                self._websocket.close()
            self._websocket = None

    def _unpack(self, payload):
        return self._msgpack_numpy.unpackb(payload, raw=False)


def _session_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    timestamp = _dt.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results_root = Path(args.results_dir).expanduser().resolve()
    ungraded_root = Path(args.ungraded_results_dir).expanduser().resolve()
    eval_results_dir = results_root / timestamp
    ungraded_session_dir = ungraded_root / timestamp
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    ungraded_session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {eval_results_dir}")
    return eval_results_dir, ungraded_session_dir


def _print_metadata(metadata: dict[str, Any]) -> None:
    def fallback(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    print("Server metadata:")
    print(json.dumps(metadata, indent=2, default=fallback))


def get_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    gear_root = Path(os.environ.get("GEAR_DROID_ROOT", DEFAULT_GEAR_DROID_ROOT))

    parser = argparse.ArgumentParser(
        description="Evaluate a MolmoSpaces/MolmoBot websocket policy on gear_droid hardware."
    )
    parser.add_argument("--gear-droid-root", type=Path, default=gear_root)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--task", default=None, help="Instruction. If omitted, prompt before rollout.")
    parser.add_argument(
        "--exo-camera",
        "--external-camera",
        dest="external_camera",
        choices=["left", "right"],
        default="left",
        help="Exterior DROID camera to map into MolmoBot's exo_camera_1 input.",
    )
    parser.add_argument("--stereo-side", default="left")
    parser.add_argument("--left-camera-id", default=_default_camera_id("left"))
    parser.add_argument("--right-camera-id", default=_default_camera_id("right"))
    parser.add_argument("--wrist-camera-id", default=_default_camera_id("wrist"))
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--max-timesteps", type=int, default=4000)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-delta", type=float, default=None)
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-zed-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open ZED cameras with depth disabled; MolmoBot uses RGB only.",
    )
    parser.add_argument("--merge-videos-on-session-end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rclone-eval-root", default="")
    parser.add_argument("--eval-exp-name", default="")
    parser.add_argument("--results-dir", default=str(repo_root / "robot_eval" / "eval_results" / "molmospaces"))
    parser.add_argument(
        "--ungraded-results-dir",
        default=str(repo_root / "robot_eval" / "ungraded_eval_results" / "molmospaces"),
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if os.environ.get("GEAR_DROID_SYSTEM") is None:
        raise ValueError(
            "GEAR_DROID_SYSTEM is required by gear_droid. Set it before running, "
            "for example: export GEAR_DROID_SYSTEM=1"
        )
    _require_camera_ids(args)
    _install_gear_droid_paths(args.gear_droid_root)
    _check_camera_runtime(args)

    _configure_zed_params(args)

    from droid.robot_env import RobotEnv
    from recorder.metadata import json_sanitize
    from recorder.session import EvalVideoSession
    from recorder.unified_launch_context import build_extra_metadata_with_launch_context
    from rollout_utils import (
        check_instruction_change,
        check_pause,
        check_reset,
        prevent_keyboard_interrupt,
        start_keyboard_listener,
        stop_keyboard_listener,
        validate_video_images,
    )

    try:
        import tqdm
    except ImportError as exc:
        raise ImportError("Install tqdm in the robot-control environment to run this script.") from exc

    policy_client = MolmoSpacesPolicyClient(
        args.policy_host,
        args.policy_port,
        timeout=args.connect_timeout,
    )
    _print_metadata(policy_client.metadata)
    print(
        "MolmoBot model inputs: "
        f"exo_camera_1 <- {args.external_camera}_image, "
        "wrist_camera <- wrist_image. "
        "All three DROID cameras are still displayed/recorded."
    )

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the gear_droid RobotEnv.")

    eval_results_dir, ungraded_session_dir = _session_dirs(args)
    eval_video = None
    if args.record:
        eval_video = EvalVideoSession(
            env,
            ungraded_session_dir,
            left_camera_id=args.left_camera_id,
            right_camera_id=args.right_camera_id,
            wrist_camera_id=args.wrist_camera_id,
            stereo_side=args.stereo_side,
        )

    instruction = args.task
    first_rollout = True
    try:
        while True:
            if instruction is None:
                instruction = input("Enter instruction: ").strip()
            elif not first_rollout and input("Change instruction? (enter y or n) ").strip().lower() == "y":
                instruction = input("Enter instruction: ").strip()

            policy_client.reset()
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

            if eval_video is not None:
                eval_video.start_rollout_recording(instruction)

            print(
                "Running rollout... press 's' to pause/play, 'r' to reset, "
                "'i' to change instruction, Ctrl+C to stop early."
            )
            start_keyboard_listener()

            obs_times = deque(maxlen=50)
            server_times = deque(maxlen=50)
            action_count = 0
            rollout_start = time.time()
            bar = tqdm.tqdm(range(args.max_timesteps))

            try:
                for t_step in bar:
                    step_start = time.time()
                    check_pause()

                    if check_reset():
                        print("[CONTROL] Resetting arm position.")
                        env.reset(randomize=False)
                        policy_client.reset()
                        print("[CONTROL] Reset complete.")
                        continue

                    if check_instruction_change():
                        stop_keyboard_listener()
                        time.sleep(0.15)
                        new_instruction = input("[CONTROL] Enter new instruction: ").strip()
                        if new_instruction:
                            instruction = new_instruction
                            policy_client.reset()
                            print(f"[CONTROL] Instruction updated to: {instruction!r}")
                        start_keyboard_listener()
                        continue

                    obs_start = time.time()
                    curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=t_step == 0)
                    obs_times.append(time.time() - obs_start)

                    request_data = _build_molmospaces_request(args, curr_obs, instruction)
                    validate_video_images(
                        exo_camera_1=request_data["exo_camera_1"],
                        wrist_camera=request_data["wrist_camera"],
                    )

                    server_start = time.time()
                    with prevent_keyboard_interrupt():
                        action_dict = policy_client.infer(request_data)
                    server_times.append(time.time() - server_start)

                    command = _action_to_robot_command(args, curr_obs, action_dict)

                    if args.display:
                        cv2 = _get_cv2()
                        left = _resize_rgb(curr_obs["left_image"], args.input_height, args.input_width)
                        wrist = _resize_rgb(curr_obs["wrist_image"], args.input_height, args.input_width)
                        right = _resize_rgb(curr_obs["right_image"], args.input_height, args.input_width)
                        combined = np.concatenate([left, wrist, right], axis=1)
                        cv2.imshow("Camera Views", combined[..., ::-1])
                        cv2.waitKey(1)

                    env.step(command)
                    action_count += 1

                    elapsed = time.time() - step_start
                    control_dt = 1.0 / DROID_CONTROL_FREQUENCY
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)

                    avg_obs_ms = np.mean(obs_times) * 1000 if obs_times else 0
                    avg_server_ms = np.mean(server_times) * 1000 if server_times else 0
                    actions_per_sec = action_count / max(time.time() - rollout_start, 1e-6)
                    timing = action_dict.get("server_timing", {})
                    total_ms = timing.get("total_ms", "?")
                    bar.set_description(
                        f"Obs: {avg_obs_ms:.1f}ms | Server: {avg_server_ms:.1f}ms "
                        f"| Infer: {total_ms}ms | Actions/sec: {actions_per_sec:.2f}"
                    )
            except KeyboardInterrupt:
                print("\nRollout interrupted.")
            finally:
                stop_keyboard_listener()
                if eval_video is not None:
                    eval_video.stop_rollout_recording()

            with suppress(Exception):
                env.reset(randomize=False)

            another = input(
                "Run another rollout in this session? [Y/n] "
                "(n = conclude session, merge per-camera videos, exit): "
            ).strip().lower()
            if another in ("n", "no"):
                if eval_video is not None:
                    eval_video.conclude_session(
                        merge=args.merge_videos_on_session_end,
                        rclone_eval_root=args.rclone_eval_root,
                        eval_exp_name=args.eval_exp_name,
                        policy_type_slug="molmospaces",
                        extra_metadata=build_extra_metadata_with_launch_context(
                            {
                                "policy_config": json_sanitize(vars(args)),
                                "server_metadata": json_sanitize(policy_client.metadata),
                                "embodiment_tag": "gear_droid_franka",
                            }
                        ),
                    )
                print("Session concluded.")
                break
            first_rollout = False
    finally:
        stop_keyboard_listener()
        policy_client.close()
        if args.display:
            cv2 = _get_cv2()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
