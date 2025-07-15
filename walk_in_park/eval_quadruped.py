

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gym
import numpy as np

from env_utils import make_mujoco_env
from wrappers import wrap_gym
from typing import Optional
import dataclasses

import collections
from crossformer.model.crossformer_model import CrossFormerModel
import jax
import pathlib
import logging
import tyro


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


@dataclasses.dataclass
class Args:
    # Minimal parameters for random policy run
    ENV_NAME: str = 'A1Run-v0'
    SEED: int = 42
    CONTROL_FREQUENCY: int = 20
    ACTION_FILTER_HIGH_CUT: Optional[float] = None
    ACTION_HISTORY: int = 1
    MAX_STEPS: int = 1000  # Maximum number of steps per episode
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    model_path: str = "hf://rail-berkeley/crossformer"
    model_step: Optional[int] = None # Finetunung step to load.
    primary_resize: int = 224
    wrist_resize: int = 128
    horizon: int = 5
    pred_horizon: int = 4
    exp_weight: int = 0

    execute_all_actions: bool = False  # Whether to execute all actions in the action chunk
    ensemble: bool = False  # Whether to use ensemble inference

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    head_name: str = "quadruped"
    dataset_name: str = "walk_in_park"
    task_suite_name: str = (
        "walk_in_park"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    action_dim: int = 7
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/quadruped/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    log_file_path: str = "data/quadruped/eval_quad.log"  # Path to save logs

def eval_quadruped(args: Args):
    """
    Evaluates the quadruped environment with the specified model and parameters.
    """
    
    
    ##################################################
    # Set up model
    ##################################################

    rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
    model = CrossFormerModel.load_pretrained(args.model_path, step=args.model_step)

    #proprio_normalization_statistics = model.dataset_statistics["proprio_single"] if "proprio_single" in model.dataset_statistics else None
    proprio_normalization_statistics = model.dataset_statistics["go1"]
    unnormalization_statistics = model.dataset_statistics["go1"]['action']
 
    ##################################################
    # Create the environment
    ##################################################
    env = make_mujoco_env(
        args.ENV_NAME,
        control_frequency=args.CONTROL_FREQUENCY,
        action_filter_high_cut=args.ACTION_FILTER_HIGH_CUT,
        action_history=args.ACTION_HISTORY)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    env = gym.wrappers.RecordVideo(
        env,
        f'videos/random_policy_{args.ACTION_FILTER_HIGH_CUT}',
        episode_trigger=lambda x: True)
    env.seed(args.seed)

    episode_return = 0
    episode_length = 0


    for i in range(args.num_trials_per_task): 
        observation, done = env.reset(), False

        history = collections.deque(maxlen=args.horizon)
        num_obs = 0
        act_queue = collections.deque(maxlen=args.pred_horizon)

        # Initialize observation_cue: shape (1, 5, 59) and timestep_mask_cue: shape (1, 5)
        obs_dim = 59
        observation_cue = np.zeros((1, 5, obs_dim), dtype=np.float32)
        timestep_mask_cue = np.zeros((1, 5), dtype=np.float32)

        for step in range(args.MAX_STEPS):
            # Pad observation to 59 dims: first 46 from observation, rest zeros
            padded_obs = np.zeros(obs_dim, dtype=np.float32)
            obs_flat = observation.flatten() if hasattr(observation, 'flatten') else np.array(observation).flatten()
            padded_obs[:46] = obs_flat[:46]

            # Update observation_cue and timestep_mask_cue as queues
            observation_cue = np.roll(observation_cue, -1, axis=1)
            observation_cue[0, -1] = padded_obs
            timestep_mask_cue = np.roll(timestep_mask_cue, -1, axis=1)
            timestep_mask_cue[0, -1] = 1.0

            element = {
                "proprio_quadruped": observation,
            }
            history.append(element)
            num_obs += 1
            obs = stack_and_pad(history, num_obs)
            obs['proprio_quadruped'] = observation_cue
            obs['timestep_pad_mask'] = timestep_mask_cue
            rng, key = jax.random.split(rng)
            actions = model.sample_actions(
                obs,
                {}, # no task
                unnormalization_statistics = unnormalization_statistics,
                head_name="quadruped",
                rng=rng,
            )
            action = actions[0][0]
            #action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
            if done:
                print(f"Episode return: {episode_return}, length: {episode_length}")
                observation, done = env.reset(), False
                # Reset cues after episode ends
                observation_cue = np.zeros((1, 5, obs_dim), dtype=np.float32)
                timestep_mask_cue = np.zeros((1, 5), dtype=np.float32)
                episode_return = 0
                episode_length = 0
            else:
                observation = next_observation

if __name__ == "__main__":
    args = tyro.cli(Args)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log_file_path,
        filemode='w',                                       # File to write logs to
        level=logging.INFO,                                 # Minimum log level to capture
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    )   

    eval_quadruped(args)
