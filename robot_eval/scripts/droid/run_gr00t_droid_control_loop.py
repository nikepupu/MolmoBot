#!/usr/bin/env python3
"""Launch GR00T's DROID control loop with RGB-only ZED camera init.

This wrapper keeps the control logic in the GR00T checkout, but patches the
GEAR-DROID ZED camera defaults before RobotEnv opens the cameras. The policies
only consume RGB frames; disabling depth avoids unnecessary bandwidth and the
third-camera open failures we have seen on the DROID station.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


GR00T_DROID_LOOP = Path(
    "/home/gear/Projects/gr00t/groot/control/envs/droid/droid_control_loop.py"
)

_JOINT_POSITION_POLICY_ACTIVE = False
_JOINT_POSITION_POLICY_LABEL = ""


def _limit_joint_delta(delta, max_delta):
    import numpy as np

    delta = np.asarray(delta, dtype=np.float64)
    max_delta = np.asarray(max_delta, dtype=np.float64)
    if max_delta.ndim == 0:
        max_delta = np.full(delta.shape, float(max_delta), dtype=np.float64)
    if max_delta.shape != delta.shape:
        raise ValueError(
            f"max relative joint delta shape {max_delta.shape} does not match action shape {delta.shape}"
        )
    max_delta = np.maximum(max_delta, 1e-6)
    relative_scale = np.abs(delta) / max_delta
    max_scale = float(np.max(relative_scale)) if relative_scale.size else 1.0
    if max_scale > 1.0:
        return delta / max_scale, max_scale
    return delta, max_scale


def _mark_joint_position_policy(label: str) -> None:
    global _JOINT_POSITION_POLICY_ACTIVE, _JOINT_POSITION_POLICY_LABEL
    _JOINT_POSITION_POLICY_ACTIVE = True
    _JOINT_POSITION_POLICY_LABEL = label
    print(f"Joint-position action safety enabled for {label}.")


def _auto_joint_action_mode(raw_action, current_joints, max_joint_step):
    import numpy as np

    raw_action = np.asarray(raw_action, dtype=np.float64)
    current_joints = np.asarray(current_joints, dtype=np.float64)
    absolute_step = raw_action - current_joints
    raw_mag = float(np.max(np.abs(raw_action))) if raw_action.size else 0.0
    abs_step_mag = float(np.max(np.abs(absolute_step))) if absolute_step.size else 0.0
    relative_mag_threshold = max(0.5, float(max_joint_step) * 4.0)
    absolute_step_threshold = max(0.5, float(max_joint_step) * 6.0)
    if raw_mag <= relative_mag_threshold and abs_step_mag >= absolute_step_threshold:
        return "relative"
    return "absolute"


def _patch_zed_depth(disable_zed_depth: bool) -> None:
    if not disable_zed_depth:
        return

    import pyzed.sl as sl
    from droid.camera_utils.camera_readers import zed_camera

    zed_camera.standard_params["depth_mode"] = sl.DEPTH_MODE.NONE
    zed_camera.advanced_params["depth_mode"] = sl.DEPTH_MODE.NONE
    print("ZED depth disabled for GR00T DROID hardware inference; RGB cameras only.")


def _modality_keys(config: dict, section: str) -> list[str]:
    cfg = config.get(section)
    if cfg is None:
        return []
    keys = cfg.get("modality_keys") if isinstance(cfg, dict) else getattr(cfg, "modality_keys", None)
    return [str(key) for key in (keys or [])]


def _delta_indices(config: dict, section: str) -> list[int]:
    cfg = config.get(section)
    if cfg is None:
        return []
    deltas = cfg.get("delta_indices") if isinstance(cfg, dict) else getattr(cfg, "delta_indices", None)
    return [int(delta) for delta in (deltas or [])]


def _language_lookup_key(language_key: str) -> str:
    return f"language.{language_key}".replace("language.", "")


def _image_for_video_key(video_key, left_image, right_image, wrist_image):
    if video_key == "exterior_image_1_left":
        return left_image
    if video_key in ("exterior_image_2_left", "exterior_image_2_right"):
        return right_image
    if video_key in ("wrist_image_left", "wrist_image"):
        return wrist_image
    raise KeyError(f"Do not know which DROID camera should source video key {video_key!r}")


class JointPositionRealRequestBuilder:
    """Flat GR00T real-robot request for joint-position DROID checkpoints.

    This covers checkpoints whose modality config is:
      video: exterior_image_1_left, exterior_image_2_left, wrist_image_left
      state: joint_position, gripper_position
      action: joint_position, gripper_position

    The server is ``Gr00tG1RealPolicyWrapper``, so videos must be JPEG bytes in
    nested lists, unlike the legacy DreamZero/raw-array request builder.
    """

    def __init__(self, modality_configs: dict):
        self.video_keys = _modality_keys(modality_configs, "video")
        self.state_keys = _modality_keys(modality_configs, "state")
        self.language_keys = _modality_keys(modality_configs, "language") or [
            "annotation.language.language_instruction",
            "annotation.language.language_instruction_2",
            "annotation.language.language_instruction_3",
        ]

    def create_histories(self):
        from collections import deque

        from groot.control.envs.droid.policy_request_builders import PolicyHistories

        return PolicyHistories(
            left_image_history=deque(maxlen=1),
            right_image_history=deque(maxlen=1),
            wrist_image_history=deque(maxlen=1),
            joint_position_history=deque(maxlen=1),
            gripper_position_history=deque(maxlen=1),
            eef_9d_history=None,
        )

    def append_observation(self, histories, curr_obs, left_image, right_image, wrist_image):
        histories.left_image_history.append(left_image)
        histories.right_image_history.append(right_image)
        histories.wrist_image_history.append(wrist_image)
        histories.joint_position_history.append(curr_obs["joint_position"].astype("float32"))
        histories.gripper_position_history.append(curr_obs["gripper_position"].astype("float32"))

    def build_request(self, curr_obs, left_image, right_image, wrist_image, histories, instruction):
        import cv2
        import numpy as np

        request = {}
        for video_key in self.video_keys:
            image = _image_for_video_key(video_key, left_image, right_image, wrist_image)
            ok, buf = cv2.imencode(".jpg", image.astype(np.uint8, copy=False))
            if not ok:
                raise RuntimeError(f"Failed to JPEG-encode video key {video_key!r}")
            request[f"video.{video_key}"] = [[buf.tobytes()]]

        for state_key in self.state_keys:
            if state_key == "joint_position":
                value = np.asarray(curr_obs["joint_position"], dtype=np.float32).reshape(1, 1, 7)
            elif state_key == "gripper_position":
                value = np.asarray(curr_obs["gripper_position"], dtype=np.float32).reshape(1, 1, 1)
            else:
                raise KeyError(f"Do not know how to source state key {state_key!r}")
            request[f"state.{state_key}"] = value

        lang = [instruction]
        for language_key in self.language_keys:
            request[language_key] = lang
            request[_language_lookup_key(language_key)] = lang
        return request


def _is_joint_position_real_schema(config: dict) -> bool:
    video_keys = set(_modality_keys(config, "video"))
    state_keys = set(_modality_keys(config, "state"))
    action_keys = set(_modality_keys(config, "action"))
    video_deltas = _delta_indices(config, "video")
    supported_video = {
        "exterior_image_1_left",
        "exterior_image_2_left",
        "exterior_image_2_right",
        "wrist_image_left",
        "wrist_image",
    }
    return (
        {"exterior_image_1_left", "wrist_image_left"} <= video_keys
        and video_keys <= supported_video
        and state_keys == {"joint_position", "gripper_position"}
        and action_keys == {"joint_position", "gripper_position"}
        and video_deltas == [0]
    )


def _patch_request_builder_detection(loop_module) -> None:
    original = loop_module._create_request_builder_from_server

    def patched(policy_client, args):
        config = policy_client.get_modality_config()
        if _is_joint_position_real_schema(config):
            _mark_joint_position_policy("GR00T DROID joint_position schema")
            print(
                "Policy mode: GR00T real-robot DROID joint_position "
                f"video_keys={_modality_keys(config, 'video')}, "
                f"state_keys={_modality_keys(config, 'state')}."
            )
            return JointPositionRealRequestBuilder(config)
        builder = original(policy_client, args)
        relative_builder_types = (
            loop_module.Gr00tRequestBuilder,
            loop_module.N17eefRelative,
            loop_module.N2RequestBuilder,
        )
        if isinstance(builder, relative_builder_types):
            _mark_joint_position_policy(type(builder).__name__)
        return builder

    loop_module._create_request_builder_from_server = patched


def _patch_joint_position_execution(
    loop_module,
    *,
    joint_action_mode: str,
    max_joint_step: float,
    debug_steps: int,
) -> None:
    if joint_action_mode == "raw":
        print("Joint-position action safety disabled by CLI flag.")
        return

    original_robot_env = loop_module.RobotEnv

    class SafeJointPositionRobotEnv(original_robot_env):
        _joint_position_debug_count = 0

        def step(self, action):
            import numpy as np

            if (
                _JOINT_POSITION_POLICY_ACTIVE
                and self.action_space == "joint_position"
                and len(action) == 8
            ):
                action = np.asarray(action, dtype=np.float64).copy()
                raw_joint_action = action[:-1].copy()
                robot_state, _timestamp = self.get_state()
                current_joints = np.asarray(robot_state["joint_positions"], dtype=np.float64)
                mode = joint_action_mode
                if mode == "auto":
                    mode = _auto_joint_action_mode(raw_joint_action, current_joints, max_joint_step)
                if mode == "relative":
                    target_joints = current_joints + raw_joint_action
                elif mode == "absolute":
                    target_joints = raw_joint_action
                else:
                    raise ValueError(
                        f"Unknown joint action mode {joint_action_mode!r}; expected auto, absolute, relative, or raw."
                    )

                raw_target_delta = target_joints - current_joints
                limited_target_delta, max_scale = _limit_joint_delta(raw_target_delta, max_joint_step)
                action[:-1] = current_joints + limited_target_delta

                if self._joint_position_debug_count < debug_steps or max_scale > 1.0:
                    print(
                        "[joint-position] "
                        f"policy={_JOINT_POSITION_POLICY_LABEL} "
                        f"mode={mode} "
                        f"raw={np.array2string(raw_joint_action, precision=4, suppress_small=True)} "
                        f"current={np.array2string(current_joints, precision=4, suppress_small=True)} "
                        f"raw_target_delta={np.array2string(raw_target_delta, precision=4, suppress_small=True)} "
                        f"limited_delta={np.array2string(limited_target_delta, precision=4, suppress_small=True)} "
                        f"target={np.array2string(action[:-1], precision=4, suppress_small=True)} "
                        f"scale={max_scale:.3f}"
                    )
                self._joint_position_debug_count += 1

            return super().step(action)

    loop_module.RobotEnv = SafeJointPositionRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--disable-zed-depth",
        dest="disable_zed_depth",
        action="store_true",
        default=True,
        help="Disable ZED depth during camera open. Enabled by default.",
    )
    parser.add_argument(
        "--no-disable-zed-depth",
        dest="disable_zed_depth",
        action="store_false",
        help="Leave the GR00T/GEAR-DROID camera defaults unchanged.",
    )
    parser.add_argument(
        "--joint-action-mode",
        choices=("auto", "absolute", "relative", "raw"),
        default="auto",
        help=(
            "How to interpret policy joint_position outputs before sending to GEAR-DROID. "
            "auto trusts absolute targets unless the output looks like a relative delta; "
            "raw disables safety conversion and limiting."
        ),
    )
    parser.add_argument(
        "--relative-joint-actions",
        dest="joint_action_mode",
        action="store_const",
        const="relative",
        help="Compatibility flag: force policy joint_position outputs to be treated as relative deltas.",
    )
    parser.add_argument(
        "--no-relative-joint-actions",
        dest="joint_action_mode",
        action="store_const",
        const="absolute",
        help="Compatibility flag: force policy joint_position outputs to be treated as absolute targets.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.08,
        help="Maximum per-step target-vs-current joint motion in radians before uniform scaling.",
    )
    parser.add_argument(
        "--joint-debug-steps",
        type=int,
        default=8,
        help="Print joint action interpretation details for the first N hardware steps.",
    )
    args, remaining = parser.parse_known_args()

    if not GR00T_DROID_LOOP.is_file():
        raise FileNotFoundError(f"GR00T DROID loop not found: {GR00T_DROID_LOOP}")

    _patch_zed_depth(args.disable_zed_depth)

    import tyro
    from groot.control.envs.droid import droid_control_loop as loop

    _patch_request_builder_detection(loop)
    _patch_joint_position_execution(
        loop,
        joint_action_mode=args.joint_action_mode,
        max_joint_step=args.max_joint_step,
        debug_steps=args.joint_debug_steps,
    )

    if not os.path.isfile(loop._DROID_CONFIG_PATH):
        raise FileNotFoundError(
            f"droid_config.yaml not found at {loop._DROID_CONFIG_PATH}. "
            "Create it with left_camera_id, right_camera_id, wrist_camera_id."
        )
    config = loop._load_droid_config()
    for key in ("left_camera_id", "right_camera_id", "wrist_camera_id"):
        if not config.get(key):
            raise RuntimeError(
                f"droid_config.yaml must set '{key}'. Edit {loop._DROID_CONFIG_PATH} "
                "with your ZED camera serials."
            )

    sys.argv = [str(GR00T_DROID_LOOP), *remaining]
    default_args = loop.Args(
        left_camera_id=config["left_camera_id"],
        right_camera_id=config["right_camera_id"],
        wrist_camera_id=config["wrist_camera_id"],
    )
    parsed_args = tyro.cli(loop.Args, default=default_args)
    loop.main(parsed_args)


if __name__ == "__main__":
    main()
