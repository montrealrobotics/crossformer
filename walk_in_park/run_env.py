

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gym
import numpy as np

from env_utils import make_mujoco_env
from wrappers import wrap_gym

# Minimal parameters for random policy run
ENV_NAME = 'A1Run-v0'
SEED = 42
CONTROL_FREQUENCY = 20
ACTION_FILTER_HIGH_CUT = None
ACTION_HISTORY = 1

env = make_mujoco_env(
    ENV_NAME,
    control_frequency=CONTROL_FREQUENCY,
    action_filter_high_cut=ACTION_FILTER_HIGH_CUT,
    action_history=ACTION_HISTORY)

env = wrap_gym(env, rescale_actions=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

env = gym.wrappers.RecordVideo(
    env,
    f'videos/random_policy_{ACTION_FILTER_HIGH_CUT}',
    episode_trigger=lambda x: True)
env.seed(SEED)

observation, done = env.reset(), False
episode_return = 0
episode_length = 0

while True:
    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(action)
    episode_return += reward
    episode_length += 1
    if done:
        print(f"Episode return: {episode_return}, length: {episode_length}")
        observation, done = env.reset(), False
        episode_return = 0
        episode_length = 0
    else:
        observation = next_observation