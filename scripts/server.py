from dataclasses import dataclass
import json_numpy

from collections import deque
import time
import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import jax
import numpy as np
import tensorflow as tf
import uvicorn

from crossformer.model.crossformer_model import CrossFormerModel
json_numpy.patch()


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


def resize(img, size=(224, 224)):
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def stack_and_pad(history: deque, num_obs: int):
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


# --- Environment Configs ---
@dataclass
class EnvironmentConfig:
    head_name: str
    dataset_name: str
    action_dim: int
    pred_horizon: int
    exp_weight: float
    horizon: int


@dataclass
class LiberoConfig(EnvironmentConfig):
    head_name: str = "single_arm"
    dataset_name: str = "bridge_dataset"
    action_dim: int = 7
    pred_horizon: int = 4
    exp_weight: float = 0
    horizon: int = 5


@dataclass
class AlohaConfig(EnvironmentConfig):
    head_name: str = "bimanual"
    dataset_name: str = "aloha_pen_uncap_diverse_dataset"
    action_dim: int = 14
    pred_horizon: int = 100
    exp_weight: float = 0
    horizon: int = 5


ENV_CONFIGS = {
    "libero": LiberoConfig,
    "aloha": AlohaConfig,
}


class HttpServer:
    def __init__(self, paths, env_config: EnvironmentConfig):
        self.models = dict()
        for name, path, step in paths:
            self.models[name] = CrossFormerModel.load_pretrained(path, step=step)

        # Use environment config
        self.head_name = env_config.head_name
        self.dataset_name = env_config.dataset_name
        self.action_dim = env_config.action_dim
        self.pred_horizon = env_config.pred_horizon
        self.exp_weight = env_config.exp_weight
        self.horizon = env_config.horizon
        self.text = None
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.resize_size = 224

        self.reset_history()

        # trigger compilation
        for name in self.models.keys():
            payload = {
                "text": "",
                "model": name,
            }
            self.reset(payload)
            payload = {
                "observation": {
                    "image_primary": np.zeros((224, 224, 3)),
                },
                "modality": "l",
                "ensemble": True,
                "model": name,
                "dataset_name": self.dataset_name,
            }
            for _ in range(self.horizon):
                start = time.time()
                print(self.sample_actions(payload))
                print(time.time() - start)

        self.reset_history()

    def run(self, port=8000, host="0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query")(self.sample_actions)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: Dict[Any, Any]):
        model_name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_img = resize(
                payload["goal"]["image_primary"],
                size=(self.resize_size, self.resize_size),
            )
            goal = {"image_primary": goal_img[None]}
            self.task = self.models[model_name].create_tasks(goals=goal)
        elif "text" in payload:
            text = payload["text"]
            self.text = text
            self.task = self.models[model_name].create_tasks(texts=[text])
        else:
            raise ValueError

        self.reset_history()

        return "reset"

    def sample_actions(self, payload: Dict[Any, Any]):
        try:
            model_name = payload.get("model", "crossformer")

            obs = payload["observation"]
            for key in obs:
                if "image" in key:
                    obs[key] = resize(
                        obs[key], size=(self.resize_size, self.resize_size)
                    )
                # normalize proprioception expect for bimanual proprioception
                if "proprio" in key and not key == "proprio_bimanual":
                    proprio_normalization_statistics = self.models[
                        model_name
                    ].dataset_statistics[self.dataset_name][key]
                    obs[key] = (obs[key] - proprio_normalization_statistics["mean"]) / (
                        proprio_normalization_statistics["std"]
                    )

            self.history.append(obs)
            self.num_obs += 1
            obs = stack_and_pad(self.history, self.num_obs)

            # add batch dim
            obs = jax.tree_map(lambda x: x[None], obs)

            unnormalization_statistics = self.models[model_name].dataset_statistics[
                self.dataset_name
            ]["action"]

            self.rng, key = jax.random.split(self.rng)
            actions = self.models[model_name].sample_actions(
                obs,
                self.task,
                unnormalization_statistics,
                head_name=self.head_name,
                rng=key,
            )[0, :, : self.action_dim]

            actions = np.array(actions)

            # whether to temporally ensemble the action predictions or return the full chunk
            if not payload.get("ensemble", True):
                print(actions)
                return json_response(actions)

            self.act_history.append(actions[: self.pred_horizon])
            num_actions = len(self.act_history)

            # select the predicted action for the current step from the history of action chunk predictions
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history
                    )
                ]
            )

            # more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.exp_weight * np.arange(num_actions))
            weights = weights / weights.sum()
            # compute the weighted average across all predictions for this timestep
            action = np.sum(weights[:, None] * curr_act_preds, axis=0)

            print(action)
            return json_response(action)
        except:
            print(traceback.format_exc())
            return "error"


def main():
    import argparse

    tf.config.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host to run on", default="0.0.0.0", type=str)
    parser.add_argument("--port", help="Port to run on", default=8000, type=int)
    parser.add_argument(
        "--env_config",
        help="Environment config name (libero, aloha)",
        default="libero",
        type=str,
    )
    args = parser.parse_args()

    # name, path, step
    paths = [
        ("crossformer", "hf://rail-berkeley/crossformer", None),
    ]

    env_config_name = args.env_config.lower()
    if env_config_name not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown env_config '{env_config_name}'. Available: {list(ENV_CONFIGS.keys())}"
        )
    env_config = ENV_CONFIGS[env_config_name]()

    server = HttpServer(paths, env_config)
    server.run(args.port, args.host)


if __name__ == "__main__":
    main()
