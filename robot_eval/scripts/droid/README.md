# Getting Started - Franka DROID

## Software installation

### On your NUC

If you already have the [DROID](https://droid-dataset.github.io) framework installed, skip this step! Otherwise, follow [these instructions](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-python-virtual-environment-conda). If you wish, you can install only [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis) instead of DROID.

### On your Inference PC

First, install the [ZED SDK](https://www.stereolabs.com/developers/release), matching your CUDA version. Then:

```bash
# Create fresh conda env for polymetis
conda create -n molmobot python=3.8
conda activate molmobot
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis

# Install ZED SDK python bindings
cd /usr/local/zed
python get_python_api.py

# Install dependencies
pip install git+https://github.com/allenai/ai2_robot_infra.git#egg=ai2_robot_infra[zed]
```


## Policy Evaluation

### Hardware Setup

Modify the [eval config](/robot_eval/config/droid.yaml) to match your hardware setup. In particular, edit:
 * `robot.robot_host` to be the IP of the NUC
 * `robot.cameras.wrist_camera.id` to be the serial number of your wrist camera (ZED Mini)
 * `robot.cameras.exo_camera_1.id` to be the serial number of your exo camera (ZED 2/2i)

If you wish, you can also enable W&B support (upload rollout videos by filling out the `wandb` block of the config and setting `wandb.enabled` to `true`).

### Policy Setup

Each MolmoBot policy class maintains its own python package, so you can install and use what you need separately.
To install and run the policy server for each policy class, see the corresponding documentation:
 - [MolmoBot](/MolmoBot/README.md)
 - [MolmoBot-Pi0](/MolmoBot-Pi0/README.md)
 - [MolmoBot-SPOC](/MolmoBot-SPOC/README.md)

### Running your policy

1. Log into Franka Desk to unlock joints and enable FCI.
2. Start the robot and gripper servers on the NUC (see [polymetis docs](https://facebookresearch.github.io/fairo/polymetis/usage.html)). For example, if DROID is installed on the NUC:
    ```bash
    # In terminal 1
    ssh droid  # ssh into the NUC
    conda activate polymetis-local
    cd droid/droid/fairo/polymetis
    launch_robot.py robot_client=franka_hardware
    ```

    ```bash
    # In terminal 2
    ssh droid  # ssh into the NUC
    conda activate polymetis-local
    cd droid/droid/fairo/polymetis
    launch_gripper.py gripper=robotiq_2f
    ```
3. Start up your policy server in terminal 3. See policy documentation for more information. For example, to run MolmoBot-DROID:
    ```bash
    # In terminal 3
    cd MolmoBot/MolmoBot
    source .venv/bin/activate
    PYTHONPATH=. python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
    ```
4. Activate the environment and control the robot.
    When running the script, the robot will first go to a home position, and then prompt the user to press enter to begin the episode.
    ```bash
    # In terminal 4
    conda activate molmobot
    python scripts/droid/run_policy.py task="put the red mug in the black bowl"
    ```

    NOTE: if you get `ffmpeg`/`ImageIO` errors, you may need to run `conda remove --force ffmpeg` and install it system-wide with `sudo apt install ffmpeg`.

### Running MolmoSpaces/MolmoBot checkpoints with gear_droid

The gear_droid-based runner reuses `RobotEnv`, keyboard controls, and per-camera session recording from `/home/gear/Projects/gr00t/external_dependencies/gear_droid`, while talking to the MolmoBot websocket server.

Start the policy server from the MolmoBot package environment:

```bash
cd /home/gear/Projects/MolmoBot/MolmoBot
source .venv/bin/activate
PYTHONPATH=. python launch_scripts/serve_molmo.py \
  --local-path ckpts/molmobot/MolmoBot-DROID \
  --action-type joint_pos
```

Then run the hardware client from the robot-control environment that has `gear_droid`, Polymetis, and camera dependencies:

```bash
export GEAR_DROID_SYSTEM=1  # or 2, ut_austin, gatech
cd /home/gear/Projects/gr00t/external_dependencies/gear_droid
python3 /home/gear/Projects/MolmoBot/robot_eval/scripts/droid/run_molmospaces_policy_gear.py \
  --task "put the red mug in the black bowl" \
  --exo-camera left \
  --policy-host localhost \
  --policy-port 8000
```

If this shell has `uv` but not a usable `python3` environment, run from the gear_droid project so `uv` resolves gear_droid's robot-control dependencies. If OpenCV fails with `numpy.core.multiarray failed to import` or `cannot import name 'aruco' from 'cv2'`, repair the `uv` environment once:

```bash
cd /home/gear/Projects/gr00t/external_dependencies/gear_droid
uv pip uninstall -y opencv-python opencv-contrib-python
uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' msgpack-numpy websockets
uv run --no-sync python - <<'PY'
import numpy
import cv2
print("numpy", numpy.__version__)
print("cv2", cv2.__version__, "aruco", hasattr(cv2, "aruco"))
PY
```

Hardware inference also needs the ZED Python API in the same environment. Action replay does not need this because it does not read live camera observations.

```bash
cd /home/gear/Projects/gr00t/external_dependencies/gear_droid
uv run --no-sync python /usr/local/zed/get_python_api.py --path /tmp/zed_py
uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' --reinstall
uv run --no-sync python - <<'PY'
import pyzed.sl as sl
print("pyzed", sl.Camera)
PY
```

If the helper downloads a wheel but fails with `No module named pip`, install the downloaded wheel directly:

```bash
cd /home/gear/Projects/gr00t/external_dependencies/gear_droid
uv pip install /home/gear/pyzed-5.2-cp311-cp311-linux_x86_64.whl
uv pip install 'numpy<2' 'opencv-contrib-python==4.6.0.66' --reinstall
```

Then run inference without syncing the project again:

```bash
export GEAR_DROID_SYSTEM=1
cd /home/gear/Projects/gr00t/external_dependencies/gear_droid
uv run --no-sync \
  python /home/gear/Projects/MolmoBot/robot_eval/scripts/droid/run_molmospaces_policy_gear.py \
    --task "put the red mug in the black bowl" \
    --exo-camera left \
    --policy-host localhost \
    --policy-port 8000
```

If that environment is missing the MolmoBot websocket client dependencies, install:

```bash
pip install msgpack-numpy websockets
```

During rollout, press `s` to pause/resume, `r` to reset the arm, `i` to change the instruction, or `Ctrl+C` to stop the current rollout. Videos are written under `robot_eval/ungraded_eval_results/molmospaces/` and merged at session end.

### Running the GR00T N1.7 MolmoBot checkpoint

The GR00T DROID stack in `/home/gear/Projects/gr00t/groot/control/envs/droid` should be used for N1.7 checkpoints. It reads the server modality config, builds the request from the checkpoint schema, and executes the returned `action.joint_position` through `RobotEnv(action_space="joint_position")`.

Copy the checkpoint from osmo into the local checkpoint folder:

```bash
mkdir -p /home/gear/Projects/MolmoBot/ckpts/gr00t/n17-molmobot-paper8-250k-scratch-wandb-ah24-h10001-20260609-1933
rsync -aP \
  osmo:/mnt/amlfs-07/shared/checkpoints/stegong/molmobot-n17/n17-molmobot-paper8-250k-scratch-wandb-ah24-h10001-20260609-1933/checkpoint-90000 \
  /home/gear/Projects/MolmoBot/ckpts/gr00t/n17-molmobot-paper8-250k-scratch-wandb-ah24-h10001-20260609-1933/
```

Start the NUC robot server:

```bash
ssh NUC_1
cd ~/Projects/droid
bash scripts/server/launch_server.sh
```

Start the GR00T policy server:

```bash
cd /home/gear/Projects/gr00t/groot/control/envs/droid
uv sync --python=3.11 --extra tf451

CHECKPOINT=/home/gear/Projects/MolmoBot/ckpts/gr00t/n17-molmobot-paper8-250k-scratch-wandb-ah24-h10001-20260609-1933/checkpoint-90000
uv run --no-sync python /home/gear/Projects/MolmoBot/robot_eval/scripts/droid/inspect_gr00t_checkpoint_modality.py "$CHECKPOINT"

uv run --no-sync python -m groot.control.main.vla.run_gr00t_server \
  --embodiment-tag <EMBODIMENT_TAG_PRINTED_FOR_THE_ABSOLUTE_JOINT_CONFIG> \
  --model-path "$CHECKPOINT" \
  --port 5555 \
  --num-inference-timesteps 8
```

For `/home/gear/Projects/molmobot_checkpoint`, the inspector prints `oxe_droid_joint_position_relative`: three DROID cameras, `joint_position + gripper_position` state, and 24-step `joint_position + gripper_position` actions. Use `OXE_DROID_JOINT_POSITION_RELATIVE` as the server embodiment tag for that checkpoint.

If the shard is named with a download suffix such as `model-00001-of-00001-005.safetensors`, add the expected Hugging Face shard name once:

```bash
ln -s /home/gear/Projects/molmobot_checkpoint/model-00001-of-00001-005.safetensors \
  /home/gear/Projects/molmobot_checkpoint/model-00001-of-00001.safetensors
```

Then start the server:

```bash
cd /home/gear/Projects/gr00t/groot/control/envs/droid
uv run --no-sync python -m groot.control.main.vla.run_gr00t_server \
  --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE \
  --model-path /home/gear/Projects/molmobot_checkpoint \
  --port 5555 \
  --num-inference-timesteps 4
```

To evaluate the same checkpoint in MolmoSpaces pick-only simulation, keep the
server running in one terminal and launch the MolmoSpaces eval in another.

Terminal 1, GR00T policy server:

```bash
cd /home/gear/Projects/gr00t/groot/control/envs/droid

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync \
  python -m groot.control.main.vla.run_gr00t_server \
    --model-path /home/gear/Projects/molmobot_checkpoint \
    --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE \
    --policy-type gr00t \
    --device cuda \
    --port 5555 \
    --num-inference-timesteps 4 \
    --use-sim-policy-wrapper \
    --rtc-control-freq 15 \
    --rtc-action-horizon 24
```

Terminal 2, MolmoSpaces eval:

```bash
cd /home/gear/Projects/MolmoBot/MolmoBot
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export MLSPACES_CACHE_DIR=$PWD/.mlspaces_cache
export MLSPACES_ASSETS_DIR=$PWD/molmo_spaces_assets
export MPLCONFIGDIR=/tmp/mplconfig
export PYTHONUNBUFFERED=1

python launch_scripts/run_eval.py \
  --use_hf_resources \
  --checkpoint_path /home/gear/Projects/molmobot_checkpoint \
  --benchmark_path "$MLSPACES_CACHE_DIR/benchmarks/molmospaces-bench-v2/20260325_1/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
  --eval_config_cls olmo.eval.configure_molmo_spaces:Gr00tDroidJointPosServerFrankaThreeViewConfig \
  --task_horizon 600 \
  --max_episodes 100 \
  --output_dir eval_output/gr00t-full-droid8-threeview-grip255-pick-100
```

Run the DROID hardware client:

```bash
cd /home/gear/Projects/gr00t/groot/control/envs/droid
uv run --no-sync python /home/gear/Projects/MolmoBot/robot_eval/scripts/droid/run_gr00t_droid_control_loop.py \
  --policy-port 5555 \
  --open-loop-horizon 8 \
  --max-timesteps 600 \
  --video-history-mode 1frame \
  --vis-cameras \
  --policy-type-slug-override gr00t_n17_molmobot_jointpos
```
