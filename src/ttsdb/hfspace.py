import subprocess
import importlib.resources
import time
import requests

from gradio_client import Client, handle_file
import yaml
import librosa
import numpy as np

from ttsdb.venv import Venv
from ttsdb.env import HF_SPACE_PATH

class HfSpace:
    def __init__(self, id: str, local: bool = False, device: str = "cuda", override: bool = False):
        self.original_id = id
        self.name = id.lower().replace("/", "_")
        self.local = local
        if self.local:
            self.venv = Venv(self.name, override=override)
        self.device = device
        self.path = HF_SPACE_PATH / self.name
        if not self.path.exists() and self.local:
            self.clone()
        # check if a config yaml file exists in the python pacakage data folder (using importlib.resources)
        with importlib.resources.path("ttsdb.data", f"{self.name}.yaml") as config_path:
            self.config_path = config_path
            print(f"Config path: {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found for {self.original_id}")
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

    def clone(self):
        subprocess.run(["git", "clone", f"https://huggingface.co/spaces/{self.original_id}", self.path], check=True)

    def _wait_for_server(self, url: str, timeout: int = 60, check_interval: float = 0.5):
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
        raise TimeoutError(f"Server at {url} did not become ready within {timeout} seconds")

    def start(self):
        if self.local:
            self.venv.install(self.path / "requirements.txt")
            self.venv.run(["pip", "install", "spaces"], cwd=self.path)
            self.process = self.venv.run(["python", "app.py"], cwd=self.path)
            # Wait for the server to be ready
            self._wait_for_server("http://localhost:7860")
            self.client = Client("http://localhost:7860")
        else:
            self.client = Client(self.original_id)
    
    def stop(self):
        if self.local:
            self.venv.stop()
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __call__(self, text: str, speaker_reference: str = None, text_reference: str = None, language: str = None, **extra_kwargs) -> tuple[np.ndarray, int]:
        if not hasattr(self, 'client') or self.client is None:
            raise RuntimeError("Space must be started before calling. Use start() or use as a context manager.")
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