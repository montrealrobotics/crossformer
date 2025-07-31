"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

import contextlib
import signal

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
from widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

from cf_scripts.client import WebClientCrossFormerPolicy
from PIL import Image

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS


# custom to policy server
flags.DEFINE_string("policy_ip", "0.0.0.0", "IP address of the server running the policy")
flags.DEFINE_integer("policy_port", 8000, "Port of the server running the policy")
flags.DEFINE_integer("open_loop_horizon", 4, "Number of actions executed in open loop")


# custom to bridge_data_robot
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer("im_size", 224, "Image size")
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 400, "num timesteps")


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def resize(img, size=(224, 224)):
    img = img.astype(np.float32)
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil_resized = pil_img.resize(size[::-1], resample=Image.LANCZOS)
    result = np.array(pil_resized, dtype=np.float32)
    return np.clip(np.round(result), 0, 255).astype(np.uint8)

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(_):
    # set up the widowx client
    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["start_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=256)
    env = WidowXGym(
        widowx_client, 256, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    # load models
    policy_client = WebClientCrossFormerPolicy(FLAGS.policy_ip, FLAGS.policy_port)


    goal_image = np.zeros((256, 256, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        modality = click.prompt(
            "Language or goal image?", type=click.Choice(["l", "g"])
        )

        if modality == "g":
            raise NotImplementedError() ## client does not support images
            # if click.confirm("Take a new goal?", default=True):
            #     assert isinstance(FLAGS.goal_eep, list)
            #     _eep = [float(e) for e in FLAGS.goal_eep]
            #     goal_eep = state_to_eep(_eep, 0)
            #     widowx_client.move_gripper(1.0)  # open gripper

            #     move_status = None
            #     while move_status != WidowXStatus.SUCCESS:
            #         move_status = widowx_client.move(goal_eep, duration=1.5)

            #     input("Press [Enter] when ready for taking the goal image. ")
            #     obs = wait_for_obs(widowx_client)
            #     obs = convert_obs(obs, FLAGS.im_size)
            #     goal = obs[None]

            # # Format task for the model
            # task = model.create_tasks(goals=goal)
            # # For logging purposes
            # goal_image = goal["image_primary"][0]
            # goal_instruction = ""

        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")
            # Format task for the model
            policy_client.reset(text)
            # For logging purposes
            goal_instruction = text
            goal_image = np.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        input("Press [Enter] to start.")

        # reset env
        obs, _ = env.reset()
        print(obs)
        # exit(0)
        time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0

        actions_from_chunk_completed = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                request_data = {"image_primary": resize(obs["image_primary"], size=(FLAGS.im_size, FLAGS.im_size))}
                policy_client.send(obs)
                
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= FLAGS.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [4, 7] of 4 cartesian velocity (6) + gripper position (1)
                        forward_pass_time = time.time()                
                        pred_action_chunk = policy_client.infer(ensemble=False)
                        print("forward pass time: ", time.time() - forward_pass_time)
                    assert pred_action_chunk.shape == (4, 7)

                # Select current action to execute from chunk
                action = np.array(pred_action_chunk[actions_from_chunk_completed], dtype=np.float64)
                actions_from_chunk_completed += 1

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    print("Truncated:", truncated)
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)