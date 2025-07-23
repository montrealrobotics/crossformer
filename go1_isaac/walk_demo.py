# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates Go1 robot teleoperation using keyboard to set velocity commands
for a trained RSL-RL policy in an arena with ArUco tags.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p go1_challenge/scripts/go1_teleop_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 teleoperation with trained policy and keyboard velocity commands.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--policy",
    type=str,
    default="unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt",
    help="Path to the trained policy file.",
)

parser.add_argument(
    "--model_path",
    type=str,
    default="hf://rail-berkeley/crossformer",
    help="Path to the pretrained CrossFormer model.",
)
parser.add_argument(
    "--model_step",
    type=int,
    default=None,
    help="Step of the pretrained CrossFormer model.",
)
parser.add_argument("--visualize_tags", action="store_true", help="Visualize detected AprilTags in a separate window.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch
import carb
import cv2

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import read_file
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.devices import Se2Keyboard

from crossformer.model.crossformer_model import CrossFormerModel
import jax
import pathlib
import logging
import collections



# AprilTag detection
try:
    from pyapriltags import Detector
except ImportError:
    print("Warning: pyapriltags library not available. Install with: pip install pyapriltags")

##
# Pre-defined configs
##
from go1_challenge.isaac_sim.go1_challenge_env_cfg import Go1ChallengeSceneCfg

PKG_PATH = Path(__file__).parent.parent.parent
DEVICE = "cpu"

NAV_FREQ = 4
LOCALIZATION_FREQ = 10  # Frequency for localization updates

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)



def load_gym_env() -> ManagerBasedRLEnv:
    """Load the Gym environment with keyboard velocity commands."""
    env_cfg = Go1ChallengeSceneCfg()
    env_cfg.sim.device = DEVICE
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False

    env_cfg.episode_length_s = 60.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    simulation_app.close()


def get_camera_intrinsics(camera_cfg):
    """Extract camera intrinsic parameters from IsaacLab camera configuration."""
    # Camera parameters from CameraCfg
    focal_length = camera_cfg.spawn.focal_length  # in mm
    horizontal_aperture = camera_cfg.spawn.horizontal_aperture  # in mm
    width = camera_cfg.width
    height = camera_cfg.height

    # Convert focal length to pixels
    fx = fy = (focal_length * width) / horizontal_aperture

    # Principal point (assuming centered)
    cx = width / 2.0
    cy = height / 2.0

    # Camera intrinsic matrix
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Camera parameters tuple format for pyapriltags: (fx, fy, cx, cy)
    camera_params = (fx, fy, cx, cy)

    return camera_matrix, camera_params


def detect_apriltags(
    rgb_image: torch.Tensor,
    camera_params: tuple,
    tag_size: float = 0.2,
    tag_family: str = "tag36h11",
    visualize: bool = False,
) -> dict:
    """
    Detect AprilTags in RGB image and estimate their poses using pyapriltags.

    Args:
        rgb_image: RGB image tensor from IsaacLab camera (H, W, 3)
        camera_params: Camera parameters tuple (fx, fy, cx, cy)
        tag_size: Physical size of tags in meters
        tag_family: AprilTag family name
        visualize: Whether to show visualization window

    Returns:
        Dictionary of detected tags with poses in robot/camera frame
    """

    try:
        # --- Format images
        # Convert tensor to numpy and ensure correct format
        if isinstance(rgb_image, torch.Tensor):
            # Convert from tensor to numpy, ensure CPU and correct dtype
            image_np = rgb_image.detach().cpu().numpy()
        else:
            image_np = rgb_image

        # Ensure image is in uint8 format and correct shape
        if image_np.dtype != np.uint8:
            # Assuming image is in [0, 1] range, convert to [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        # Convert to grayscale for AprilTag detection
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Detect tags with pose estimation
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        if len(tags) == 0:
            if visualize:
                # Show image even if no tags detected
                cv2.imshow("AprilTag Detection", image_np)
                cv2.waitKey(1)
            return {}

        detected_tags = {}

        # Create visualization if requested
        if visualize:
            # Convert grayscale back to color for visualization
            vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        for tag in tags:
            tag_id = tag.tag_id

            # Get pose information
            position = tag.pose_t.flatten()  # Translation vector [x, y, z]
            rotation_matrix = tag.pose_R  # 3x3 rotation matrix

            # Calculate distance
            distance = np.linalg.norm(position)

            detected_tags[tag_id] = {
                "corners": tag.corners.tolist(),
                "center": tag.center.tolist(),
                "pose": {
                    "position": position.tolist(),  # [x, y, z] in camera frame
                    "rotation_matrix": rotation_matrix.tolist(),
                },
                "distance": float(distance),
                "confidence": float(tag.decision_margin),  # Detection confidence
            }

            # Add visualization elements
            if visualize:
                # Draw tag outline
                corners = tag.corners.astype(int)
                for idx in range(len(corners)):
                    cv2.line(vis_image, tuple(corners[idx - 1]), tuple(corners[idx]), (0, 255, 0), 2)

                # Add tag ID text
                center = tag.center.astype(int)
                cv2.putText(
                    vis_image,
                    f"ID:{tag_id}",
                    org=(center[0] - 20, center[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )

                # Add distance text
                cv2.putText(
                    vis_image,
                    f"{distance:.2f}m",
                    org=(center[0] - 20, center[1] + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                )

        # Show visualization window
        if visualize:
            cv2.imshow("AprilTag Detection", vis_image)
            cv2.waitKey(1)  # Non-blocking

        print(f"[INFO] Detected {len(detected_tags)} AprilTags: {list(detected_tags.keys())}")
        return detected_tags

    except Exception as e:
        print(f"[ERROR] AprilTag detection failed: {e}")
        return {}

def stack_and_pad(history: collections.deque, num_obs: int):
    ## copied from scripts.server.py

    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs

def main():
    """Main teleoperation loop."""

    # Setup environment
    env = load_gym_env()

    # Setup keyboard interface
    # keyboard_interface = VelocityKeyboardInterface(lin_vel_scale=0.8, ang_vel_scale=0.6)
    sensitivity_lin = 1.0  # Default sensitivity for keyboard commands
    sensitivity_ang = 1.0  # Default sensitivity for angular commands
    teleop_interface = Se2Keyboard(
        v_x_sensitivity=sensitivity_lin, v_y_sensitivity=sensitivity_lin, omega_z_sensitivity=sensitivity_ang
    )

    teleop_interface.add_callback("R", env.reset)
    teleop_interface.add_callback("ESCAPE", quit_cb)

    print(teleop_interface)

    # Load trained policy
    #policy = load_policy_rsl("unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt")  # Adjust filename


    rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
    model = CrossFormerModel.load_pretrained(args_cli.model_path, step=args_cli.model_step)
    #proprio_normalization_statistics = model.dataset_statistics["proprio_single"] if "proprio_single" in model.dataset_statistics else None
    proprio_normalization_statistics = model.dataset_statistics["go1"]
    unnormalization_statistics = model.dataset_statistics["go1"]['action']


    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Reset environment
    teleop_interface.reset()
    obs, _ = env.reset()

    count = 0
    nav_count = 1 / (NAV_FREQ * env.cfg.sim.dt)
    localization_count = 1 / (LOCALIZATION_FREQ * env.cfg.sim.dt)

    # Get camera intrinsics from environment configuration
    camera_cfg = env.cfg.scene.camera
    camera_matrix, camera_params = get_camera_intrinsics(camera_cfg)

    print(f"[INFO] Camera intrinsics calculated:")
    print(f"  Camera matrix:\n{camera_matrix}")
    print(f"  Camera params: {camera_params}")
    print(f"  Image size: {camera_cfg.width}x{camera_cfg.height}")
    if args_cli.visualize_tags:
        print(f"[INFO] Tag visualization enabled")

    print("\n[INFO] Teleoperation started. Use WASD+QE to control the robot.")
    #things for  collecting obs
    horizon = 5
    pred_horizon = 4
    history = collections.deque(maxlen=horizon)
    num_obs = 0
    act_queue = collections.deque(maxlen=pred_horizon)

    #cues
    obs_dim = 59
    observation_cue = np.zeros((1, 5, obs_dim), dtype=np.float32)
    timestep_mask_cue = np.zeros((1, 5), dtype=np.float32)
    task = model.create_tasks(texts=['walk'])

    while simulation_app.is_running():
        with torch.inference_mode():
            # Navigation loop
            if count % localization_count == 0:
                # Get camera images
                rgb_data = env.scene["camera"].data.output["rgb"][0, ..., :3]
                distance_data = env.scene["camera"].data.output["distance_to_image_plane"][0]

                # AprilTag localization using pyapriltags
                detected_tags = detect_apriltags(
                    rgb_image=rgb_data,
                    camera_params=camera_params,  # Use camera params tuple
                    tag_size=0.2,  # 0.2m tag size as specified
                    tag_family="tag36h11",
                    visualize=False,  # args_cli.visualize_tags,  # Add visualization option
                )

                if detected_tags:
                    print("-------------------------------")
                    print(f"[LOCALIZATION] Detected tags: {list(detected_tags.keys())}")
                    for tag_id, tag_data in detected_tags.items():
                        pos = tag_data["pose"]["position"]
                        dist = tag_data["distance"]
                        conf = tag_data["confidence"]
                        print(
                            f"  Tag {tag_id}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dist={dist:.2f}m, conf={conf:.2f}"
                        )
                    print("-------------------------------")

            # Teleop Commands
            #keyboard_command = teleop_interface.advance()
            #obs["policy"][:, 9:12] = torch.tensor(keyboard_command)  # Update policy observation with keyboard command

            # Policy inference
            #action = policy(obs["policy"])
            # Pad observation to 59 dims: first 46 from observation, rest zeros
            padded_obs = np.zeros(obs_dim, dtype=np.float32)
            obs = obs['policy']
            obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
            o_dims = obs_flat.shape[0]
            padded_obs[:o_dims] = obs_flat[:o_dims]

            # Update observation_cue and timestep_mask_cue as queues
            observation_cue = np.roll(observation_cue, -1, axis=1)
            observation_cue[0, -1] = padded_obs
            timestep_mask_cue = np.roll(timestep_mask_cue, -1, axis=1)
            timestep_mask_cue[0, -1] = 1.0

            element = {
                "proprio_quadruped": obs,
            }
            history.append(element)
            num_obs += 1
            obs = stack_and_pad(history, num_obs)
            obs['proprio_quadruped'] = observation_cue
            obs['timestep_pad_mask'] = timestep_mask_cue
            rng, key = jax.random.split(rng)
            actions = model.sample_actions(
                obs,
                task, #task is "walk"
                unnormalization_statistics = unnormalization_statistics,
                head_name="quadruped",
                rng=rng,
            )
            action = actions[0][0]
            #action = 
            action = np.array(action)
            action = torch.from_numpy(action).float()
            # Step environment
            print(action) 
            action = action.unsqueeze(0)
            obs, _, _, _, _ = env.step(action)

            count += 1

    # Cleanup visualization window
    if args_cli.visualize_tags:
        cv2.destroyAllWindows()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

