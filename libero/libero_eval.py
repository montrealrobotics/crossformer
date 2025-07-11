import numpy as np
import tensorflow as tf
import jax

import collections
import dataclasses
import logging
import pathlib
from typing import Optional

import imageio
import tqdm
import tyro

from libero.libero import benchmark
from libero_utils import get_libero_env, quat2axisangle
from crossformer.model.crossformer_model import CrossFormerModel


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def resize(img, size=(224, 224)):
    ## copied from scripts.server.py
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


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
    head_name: str = "single_arm"
    dataset_name: str = "liber_o10"
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    action_dim: int = 7
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    log_file_path: str = "data/libero/eval_libero.log"  # Path to save logs




def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")


    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
    model = CrossFormerModel.load_pretrained(args.model_path, step=args.model_step)
    
    proprio_normalization_statistics = model.dataset_statistics["proprio_single"] if "proprio_single" in model.dataset_statistics else None
    unnormalization_statistics = model.dataset_statistics["action"]
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Reset model spesific objects
            created_task = model.create_tasks(texts=[task_description])
            history = collections.deque(maxlen=args.horizon)
            num_obs = 0
            act_queue = collections.deque(maxlen=args.pred_horizon)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = resize(obs["agentview_image"][::-1, ::-1], size=(args.primary_resize, args.primary_resize))
                    wrist_img = resize(obs["robot0_eye_in_hand_image"][::-1, ::-1], size=(args.wrist_resize, args.wrist_resize))

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    
                    ## prepare observation and add it to history
                    element = {
                        "image_primary": img,
                        ## model does not have separate wrist image input 
                        ## or proporio input for single arms :(

                        "image_left_wrist": wrist_img,
                        "proprio_single": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }
                    if proprio_normalization_statistics is not None:
                        element["proprio_single"] = (element["proprio_single"] - proprio_normalization_statistics["mean"]) / (
                                            proprio_normalization_statistics["std"]
                                    )
                    
                    history.append(element)
                    num_obs += 1


                    if args.ensemble or (not args.execute_all_actions) or (args.execute_all_actions and not act_queue):
                        ## we want the model to predict a new action chunk
                        ## when we are using ensemble
                        ## or we are only executing the first action in the chunk
                        ## or we are executing all actions in the chunk and the action history is empty
                        obs = stack_and_pad(history, num_obs)
                        # Query model to get action
                        obs = jax.tree_map(lambda x: x[None], obs)

                        rng, key = jax.random.split(rng)
                        actions = model.sample_actions(
                            obs,
                            created_task,
                            unnormalization_statistics,
                            head_name=args.head_name,
                            rng=key,
                        )[0, :, : args.action_dim]
                        actions = np.array(actions)
                        ## for pretraining the gripper action is in [0, 1], 0: close, 1: open
                        ## therefore for finetuning we converted libero datasets gripper action from [-1, 1] -1:open, 1: close
                        ## to [0, 1] and we need to convert it back to [-1, 1] for the libero env
                        actions[:, -1] = 2 * (1 - actions[:, -1]) - 1 
                    if args.ensemble:
                        ## when using emsemble, action_queue is used to store the history of action chunk predictions
                        act_queue.append(actions[: args.pred_horizon])
                        num_actions = len(act_queue)

                        # select the predicted action for the current step from the history of action chunk predictions
                        curr_act_preds = np.stack(
                            [
                                pred_actions[i]
                                for (i, pred_actions) in zip(
                                    range(num_actions - 1, -1, -1), act_queue
                                )
                            ]
                        )

                        # more recent predictions get exponentially *less* weight than older predictions
                        weights = np.exp(-args.exp_weight * np.arange(num_actions))
                        weights = weights / weights.sum()
                        # compute the weighted average across all predictions for this timestep
                        action = np.sum(weights[:, None] * curr_act_preds, axis=0)
                    elif not args.execute_all_actions:
                        ## no temp ensemble, and not executing all actions 
                        ## so just take the first action in the action chunk
                        ## and predict a new chunk
                        action = actions[0]
                    elif args.execute_all_actions and not act_queue:
                        ## no temp ensemble, but executing all actions in predicted chunk
                        ## the action_queue is used as a action plan for next actions
                        ## same as openpi implementation
                        ## and the queue is empty 
                        ## so use the queue as the holder of actions in the chunk
                        ## and pop the first action in the queue
                        act_queue.extend(actions[: args.pred_horizon])
                        action = act_queue.popleft()
                    else:
                        ## no temp ensemble, but executing all actions in predicted chunk
                        ## the action history is used as a action queue for next actions
                        ## and the queue is empty 
                        ## same as openpi implementation
                        ## but queue is not empty
                        ## so just pop the first action in the queue
                        action = act_queue.popleft()
                            
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            ## instead of imageio and logging, use wandb
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    env.close()



if __name__ == "__main__":
    args = tyro.cli(Args)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log_file_path,
        filemode='w',                                       # File to write logs to
        level=logging.INFO,                                 # Minimum log level to capture
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    )   
    eval_libero(args)