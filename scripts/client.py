from typing import Dict, Any

from abc import ABC, abstractmethod

from absl import logging
import time
import numpy as np
import requests
import json_numpy
from json_numpy import loads

json_numpy.patch()


class WebClientPolicy(ABC):
    """Abstract interface for a web client policy."""

    @abstractmethod
    def infer(self, obs: Dict[str, Any], ensemble: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self, task: str) -> None:
        pass


class WebClientCrossFormerPolicy(WebClientPolicy):
    """Implements the WebClientPolicy interface for CrossFormer model by communicating with a web server.

    See scripts/server.py for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"http://{host}:{port}"
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()
        while True:
            try:
                resp = requests.post(f"{self._uri}/reset", json={"text": "ping"})
                if resp.status_code == 200:
                    logging.info(f"Server at {self._uri} is available.")
                    break
            except Exception as e:
                logging.info(f"Waiting for server at {self._uri}... ({e})")
            if time.time() - start_time > 60.0:
                raise TimeoutError(
                    f"Server at {self._uri} did not respond within 60 seconds."
                )
            time.sleep(5.0)

    def infer(self, ensemble: bool = False) -> np.ndarray:
        action = loads(
            requests.post(
                f"{self._uri}/query",
                json={"ensemble": ensemble},
            ).json()
        )
        return action

    def send(self, obs: Dict[str, Any]) -> None:
        """Sends an observation to the server so that it is appended to history."""
        requests.post(f"{self._uri}/append", json={"observation": obs})

    def reset(self, task: str) -> None:
        """Only supports text-based task descriptions."""
        requests.post(f"{self._uri}/reset", json={"text": task})
