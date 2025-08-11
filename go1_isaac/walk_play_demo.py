# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
# append RSL-RL cli arguments
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from crossformer.model.crossformer_model import CrossFormerModel
import jax
import pathlib
import logging
import collections



from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# import gym envs
import isaaclab_tasks  # noqa: F401
import go1_challenge.isaaclab_tasks  # noqa: F401
import numpy as np

# PLACEHOLDER: Extension template (do not remove this comment)
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
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    #agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    #log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    #log_root_path = os.path.abspath(log_root_path)
    #print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    #elif args_cli.checkpoint:
    #    resume_path = retrieve_file_path(args_cli.checkpoint)
    #else:
    #    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = "./"
    #log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    #if isinstance(env.unwrapped, DirectMARLEnv):
    #    env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    #we're...gonna assume it clips
    #env = RslRlVecEnvWrapper(env) #, clip_actions = False) #clip_actions=agent_cfg.clip_actions)

    #print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
    model = CrossFormerModel.load_pretrained(args_cli.model_path, step=args_cli.model_step)
    proprio_normalization_statistics = model.dataset_statistics["proprio_quadruped"]
    unnormalization_statistics = model.dataset_statistics['action']
    #things for  collecting obs
    horizon = 1
    pred_horizon = 4
    history = collections.deque(maxlen=horizon)
    num_obs = 0
    act_queue = collections.deque(maxlen=pred_horizon)

    #obs_dim SHOULD be 45 dimensional
    obs_dim =  proprio_normalization_statistics['mean'].shape[-1]
    observation_cue = np.zeros((1, horizon, obs_dim), dtype=np.float32)
    timestep_mask_cue = np.zeros((1, horizon), dtype=np.float32)
    task = model.create_tasks(texts=['walk'])
    prev_act = np.zeros((12), dtype=np.float32)


    # extract the neural network module

    # export policy to onnx/jit
    
    dt = env.unwrapped.step_dt

    # reset environment
    #obs, _ = env.get_observations()
    obs, _ = env.reset()

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs = obs['policy'].cpu().numpy()
            obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
            prop_dim = 45 #this is magic number
            obs_flat = obs_flat[:prop_dim]
            obs_flat = obs_flat
            #add previous action
            obs_flat = np.concatenate((obs_flat, prev_act), axis = -1)
            observation_cue = np.roll(observation_cue, -1, axis=1)

            obs_flat_norm = (obs_flat - proprio_normalization_statistics['mean']) / (proprio_normalization_statistics['std'] + 1e-8)
            observation_cue[0, -1] = obs_flat_norm#obs_flat 
            timestep_mask_cue = np.roll(timestep_mask_cue, -1, axis=1)
            timestep_mask_cue[0, -1] = 1.0

            element = {
                "proprio_quadruped": obs,
            }
            history.append(element)
            num_obs += 1

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
            action = np.array(action)
            prev_act = action  #set it before transforming
            action = torch.from_numpy(action).float()
            # Step environment
            print(action) 

            action = action.unsqueeze(0)

            action = np.array(action)
            action = torch.from_numpy(action).float()
            print(action)
            #actions = env.action_space.sample()
            #actions = torch.from_numpy(actions).float()
            # env stepping
            obs, rew, trunc, done, info = env.step(action)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()