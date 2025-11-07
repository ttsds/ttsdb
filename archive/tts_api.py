from time import sleep
import io
from functools import lru_cache
import tempfile
from pathlib import Path

import requests
import librosa
import soundfile as sf
import whisper

SYSTEMS = {
    "amphion": 8001,
}

class TTSApi():
    def __init__(self, systems=SYSTEMS, use_docker=True, whisper_model="base.en", trim_input_silence=False, trim_output_silence=True):
        self.systems = systems
        self.systems_info = {}
        self.use_docker = use_docker
        for system, port in self.systems.items():
            print(f"Starting {system} on port {port}")
            self.systems_info[system] = self.get_info(system)
            self.systems_info[system]["port"] = port
        self.whisper = whisper.load_model(whisper_model)
        self.whisper_model = whisper_model
        self.trim_input_silence = trim_input_silence
        self.trim_output_silence = trim_output_silence

    def _process_audio(self, audio):
        if isinstance(audio, tuple):
            return audio
        elif isinstance(audio, str) or isinstance(audio, Path):
            audio, sr = librosa.load(audio)
            return sr, audio

    @lru_cache
    def _create_request_url(self, system, port):
        if not self.use_docker:
            request_url = f"http://{system}:{port}"
        else:
            request_url = f"http://localhost:{port}"
        while True:
            try:
                ok = requests.get(request_url + "/ready").ok
                if ok:
                    break
            except Exception as e:
                print(f"Waiting for {system} to be ready")
                sleep(1)
        return request_url

    def set_trim_input_silence(self, trim_input_silence):
        self.trim_input_silence = trim_input_silence

    def set_trim_output_silence(self, trim_output_silence):
        self.trim_output_silence = trim_output_silence

    def get_info(self, system):
        if system in self.systems_info:
            return self.systems_info[system]
        port = self.systems[system]
        # get versions for each system
        request_url = self._create_request_url(system, port) + "/info" 
        response = requests.get(request_url)
        return response.json()

    def synthesize(self, text, system, version, input_audio, input_text=None, timeout=-1):
        info = self.get_info(system)
        byte_arr = io.BytesIO()
        input_audio = self._process_audio(input_audio)
        if self.trim_input_silence:
            audio = librosa.effects.trim(input_audio[1], top_db=20)[0]
            input_audio = (input_audio[0], audio)
        sf.write(byte_arr, input_audio[1], input_audio[0], format="wav")
        byte_arr.seek(0)
        request_url = self._create_request_url(system, info["port"]) + "/synthesize"
        data = {
            "text": text,
            "version": version,
        }
        version_idx = info["versions"].index(version)
        if info["requires_text"][version_idx]:
            if input_text is None:
                print(f"Reference text is required for {version}, generating using whisper {self.whisper_model} model")
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    sf.write(f.name, input_audio[1], input_audio[0], format="wav")
                    input_text = self.whisper.transcribe(f.name)["text"]
                print(f"Reference text: {input_text}")
        if input_text is None:
            input_text = ""
        data["speaker_txt"] = input_text
        print(input_text)
        if timeout > 0:
            response = requests.post(
                request_url,
                data=data,
                files={"speaker_wav": byte_arr},
                timeout=timeout,
            )
        else:
            response = requests.post(
                request_url,
                data=data,
                files={"speaker_wav": byte_arr},
            )
        # trim silence
        if self.trim_output_silence:
            with io.BytesIO(response.content) as f:
                audio, sr = sf.read(f)
                audio = librosa.effects.trim(audio, top_db=20)[0]
                byte_arr = io.BytesIO()
                sf.write(byte_arr, audio, sr, format="wav")
                byte_arr.seek(0)
                return byte_arr.read()
        return response.content
        