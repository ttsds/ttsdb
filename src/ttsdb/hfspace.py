import socket
import subprocess
import importlib.resources
import time
import requests

from gradio_client import Client, handle_file
import yaml
import librosa
import numpy as np

from ttsdb.abstract import BaseTTS
from ttsdb.venv import Venv
from ttsdb.env import HF_SPACE_PATH


class BaseHfSpace(BaseTTS):
    """Base class for HuggingFace Space TTS systems."""

    def __init__(self, id: str):
        self.original_id = id
        self.name = id.lower().replace("/", "_")
        self.client = None
        self._load_config()

    def _load_config(self):
        """Load the configuration yaml file from package data."""
        with importlib.resources.path("ttsdb.data", f"{self.name}.yaml") as config_path:
            self.config_path = config_path
            print(f"Config path: {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found for {self.original_id}")
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

    def synthesize(
        self,
        text: str,
        speaker_reference: str = None,
        text_reference: str = None,
        language: str = None,
        **extra_kwargs,
    ) -> tuple[np.ndarray, int]:
        if self.client is None:
            raise RuntimeError(
                "Space must be started before calling. Use start() or use as a context manager."
            )
        text_param = self.config["parameters"]["text"]
        speaker_reference_param = self.config["parameters"]["speaker_reference"]
        text_reference_param = self.config["parameters"]["text_reference"]
        language_param = self.config["parameters"]["language"]
        kwargs = {}
        default_params = self.config["default_parameters"]
        for param, value in default_params.items():
            kwargs[str(param)] = value
        if text_param is not None:
            kwargs[str(text_param)] = text
        if speaker_reference_param is not None:
            kwargs[str(speaker_reference_param)] = handle_file(speaker_reference)
        if text_reference_param is not None:
            kwargs[str(text_reference_param)] = text_reference
        if language_param is not None:
            kwargs[str(language_param)] = language
        for param, value in extra_kwargs.items():
            kwargs[str(param)] = value
        audio, sr = librosa.load(self.client.predict(**kwargs))
        return audio, sr


class OnlineHfSpace(BaseHfSpace):
    """HuggingFace Space TTS that connects to a remote hosted space."""

    def __init__(self, id: str):
        super().__init__(id)
        BaseTTS.__init__(self, f"hfspace:{id}")

    def start(self):
        self.client = Client(self.original_id)

    def stop(self):
        self.client = None


class LocalHfSpace(BaseHfSpace):
    """HuggingFace Space TTS that runs locally."""

    DEFAULT_PORT = 7860

    def __init__(
        self, id: str, device: str = "cuda", override: bool = False, port: int = None
    ):
        super().__init__(id)
        BaseTTS.__init__(self, f"hfspace-local:{id}")
        self.device = device
        self.venv = Venv(self.name, override=override)
        self.path = HF_SPACE_PATH / self.name
        self.process = None
        if port is not None:
            self.port = port
        else:
            self.port = self._find_available_port(self.DEFAULT_PORT)
        if not self.path.exists():
            self.clone()

    def clone(self):
        subprocess.run(
            [
                "git",
                "clone",
                f"https://huggingface.co/spaces/{self.original_id}",
                self.path,
            ],
            check=True,
        )

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port."""
        for offset in range(max_attempts):
            port = start_port + offset
            if not self._is_port_in_use(port):
                return port
        raise RuntimeError(
            f"Could not find an available port after {max_attempts} attempts starting from {start_port}"
        )

    def _wait_for_server(
        self, url: str, timeout: int = 60, check_interval: float = 0.5
    ):
        """Wait for the server to be ready by polling the URL."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            time.sleep(check_interval)
        raise TimeoutError(
            f"Server at {url} did not become ready within {timeout} seconds"
        )

    def start(self):
        self.venv.install(self.path / "requirements.txt")
        self.venv.run(["pip", "install", "spaces"], cwd=self.path)
        # Pass the port via GRADIO_SERVER_PORT environment variable
        env = {"GRADIO_SERVER_PORT": str(self.port)}
        self.process = self.venv.run(["python", "app.py"], cwd=self.path, env=env)
        # Wait for the server to be ready
        server_url = f"http://localhost:{self.port}"
        self._wait_for_server(server_url)
        self.client = Client(server_url)

    def stop(self):
        self.venv.stop()
        self.client = None
        self.process = None
