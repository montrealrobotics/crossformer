from typing import Dict

from absl import logging
import time
import numpy as np
import requests
import json_numpy
from json_numpy import loads

json_numpy.patch()


class WebClientPolicy:
    """Implements the Policy interface by communicating with a web server.

    See WebsocketPolicyServer for a corresponding server implementation.
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

    def infer(self, obs: Dict, ensemble: bool = False) -> np.ndarray:
        action = loads(
            requests.post(
                f"{self._uri}/query",
                json={"observation": obs, "ensemble": ensemble},
            ).json()
        )
        return action

    def reset(self, task: str) -> None:
        requests.post(f"{self._uri}/reset", json={"text": task})
