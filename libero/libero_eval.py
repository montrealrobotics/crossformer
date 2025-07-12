import collections
import dataclasses
import logging
import pathlib

import imageio
from libero.libero import benchmark
import numpy as np
from libero_utils import get_libero_env, quat2axisangle
import tqdm
import tyro
from client import WebClientCrossFormerPolicy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    execute_all_actions: bool = (
        False  # Whether to execute all actions in the action chunk
    )
    pred_horizon: int = 4  # Number of actions to predict in the action chunk
    ensemble: bool = False  # Whether to use ensemble inference

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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

    client = WebClientCrossFormerPolicy(host="0.0.0.0", port=8000)

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
            client.reset(task_description)
            num_obs = 0
            act_queue = collections.deque(maxlen=args.pred_horizon)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
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
                    img = obs["agentview_image"][::-1, ::-1]
                    wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    ## prepare observation and add it to history
                    element = {
                        "image_primary": img,
                        ## model does not have separate wrist image input
                        ## or proporio input for single arms :(
                        # "wrist_image": wrist_img,
                        # "proprio": np.concatenate(
                        #     (
                        #         obs["robot0_eef_pos"],
                        #         quat2axisangle(obs["robot0_eef_quat"]),
                        #         obs["robot0_gripper_qpos"],
                        #     )
                        # ),
                    }
                    # if proprio_normalization_statistics is not None:
                    #     element["proprio"] = (element["proprio"] - proprio_normalization_statistics["mean"]) / (
                    #                         proprio_normalization_statistics["std"]
                    #                 )

                    num_obs += 1

                    if (
                        args.ensemble
                        or (not args.execute_all_actions)
                        or (args.execute_all_actions and not act_queue)
                    ):
                        ## we want the model to predict a new action chunk
                        ## when we are using ensemble
                        ## or we are only executing the first action in the chunk
                        ## or we are executing all actions in the chunk and the action queue is empty
                        actions = client.infer(element, args.ensemble)
                        actions = np.array(actions)
                        ## for pretraining the gripper action is in [0, 1], 0: close, 1: open
                        ## therefore for finetuning we converted libero datasets gripper action from [-1, 1] -1:open, 1: close
                        ## to [0, 1] and we need to convert it back to [-1, 1] for the libero env
                        actions[:, -1] = 2 * (1 - actions[:, -1]) - 1 

                    if args.ensemble:
                        action = actions
                    if not args.ensemble and not args.execute_all_actions:
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
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
