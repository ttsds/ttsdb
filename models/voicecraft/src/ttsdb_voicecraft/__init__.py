from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import (
    AudioOutput,
    VoiceCloningTTSBase,
    get_variant_checkpoint_dir,
    setup_vendor_path,
)

setup_vendor_path("ttsdb_voicecraft")

__all__ = ["VoiceCraft"]


class VoiceCraft(VoiceCloningTTSBase):
    """VoiceCraft voice cloning TTS model."""

    _package_name = "ttsdb_voicecraft"
    SAMPLE_RATE = 16000

    def _iter_torch_modules(self):
        modules = []
        try:
            import torch

            for candidate in [getattr(self, "model", None)]:
                if isinstance(candidate, torch.nn.Module):
                    modules.append(candidate)
        except Exception:
            pass
        return modules

    def _get_align_model(self):
        if getattr(self, "_align_model", None) is None:
            from whisperx import load_align_model

            self._align_model, self._align_metadata = load_align_model(
                language_code="en", device=self.device
            )
        return self._align_model, self._align_metadata

    def _get_whisperx_model(self, model_name: str = "base.en"):
        if getattr(self, "_whisperx_model", None) is None:
            from whisperx import load_model

            asr_options = {
                "suppress_numerals": True,
                "max_new_tokens": None,
                "clip_timestamps": None,
                "hallucination_silence_threshold": None,
            }
            self._whisperx_model = load_model(model_name, self.device, asr_options=asr_options)
        return self._whisperx_model

    def _get_whisper_model(self, model_name: str = "base.en"):
        if getattr(self, "_whisper_model", None) is None:
            from whisper import load_model
            from whisper.tokenizer import get_tokenizer

            model = load_model(model_name, device=str(self.device))
            tokenizer = get_tokenizer(multilingual=False)
            suppress_tokens = [-1] + [
                i
                for i in range(tokenizer.eot)
                if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
            ]
            self._whisper_model = (model, suppress_tokens)
        return self._whisper_model

    def _load_model(self, load_path: str):
        import torch
        from data.tokenizer import AudioTokenizer, TextTokenizer

        from models import voicecraft

        base = Path(load_path)
        variant_dir = get_variant_checkpoint_dir(base, config=self.model_config)

        safetensors_path = variant_dir / "model.safetensors"
        config_path = variant_dir / "config.json"
        if safetensors_path.exists() and config_path.exists():
            self.model = voicecraft.VoiceCraft.from_pretrained(str(variant_dir))
            self.model.eval()
        else:
            checkpoint_candidates = [
                variant_dir / "830M_TTSEnhanced.pth",
                variant_dir / "giga830M.pth",
            ]
            ckpt_path = next((p for p in checkpoint_candidates if p.exists()), None)
            if ckpt_path is None:
                raise FileNotFoundError(
                    "VoiceCraft checkpoint not found. Expected one of: "
                    + ", ".join(str(p) for p in [safetensors_path, *checkpoint_candidates])
                )

            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            self.model = voicecraft.VoiceCraft(ckpt["config"])
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            if ckpt.get("phn2num") is not None:
                self.model.args.phn2num = ckpt["phn2num"]

        encodec_path = base / "shared" / "encodec" / "encodec_4cb2048_giga.th"
        if not encodec_path.exists():
            raise FileNotFoundError(f"VoiceCraft EnCodec checkpoint not found at {encodec_path}.")
        self.audio_tokenizer = AudioTokenizer(signature=str(encodec_path), device=self.device)
        self.text_tokenizer = TextTokenizer(backend="espeak")

        # Default inference settings from the original demo.
        self.silence_tokens = [1388, 1898, 131]
        self.codec_audio_sr = 16000
        self.codec_sr = 50
        self.top_k = 0
        self.top_p = 0.9
        self.temperature = 1.0
        self.kvcache = 1
        self.stop_repetition = 3
        self.sample_batch_size = 2
        self.cut_off_sec = 3.6

        return self.model

    def _replace_numbers_with_words(self, text: str) -> str:
        from num2words import num2words

        text = re.sub(r"(\d+)", r" \1 ", text)

        def replace_with_words(match):
            num = match.group(0)
            try:
                return num2words(num)
            except Exception:
                return num

        return re.sub(r"\b\d+\b", replace_with_words, text)

    def _align_transcript(self, transcript: str, audio_path: Path):
        segments = None
        try:
            from aeneas.executetask import ExecuteTask
            from aeneas.task import Task

            config_string = "task_language=eng|os_task_file_format=json|is_text_type=plain"
            tmp_transcript = audio_path.with_suffix(".txt")
            tmp_sync_map = audio_path.with_suffix(".json")

            tmp_transcript.write_text(transcript)
            task = Task(config_string=config_string)
            task.audio_file_path_absolute = str(audio_path.resolve())
            task.text_file_path_absolute = str(tmp_transcript.resolve())
            task.sync_map_file_path_absolute = str(tmp_sync_map.resolve())
            ExecuteTask(task).execute()
            task.output_sync_map_file()

            import json

            with open(tmp_sync_map) as f:
                fragments = json.load(f).get("fragments", [])

            segments = [
                {
                    "start": float(fragment["begin"]),
                    "end": float(fragment["end"]),
                    "text": " ".join(fragment.get("lines", [])),
                }
                for fragment in fragments
            ]
        except Exception:
            segments = None

        if not segments:
            import torchaudio

            info = torchaudio.info(str(audio_path))
            audio_dur = info.num_frames / info.sample_rate
            segments = [{"start": 0.0, "end": audio_dur, "text": transcript}]

        align_model, align_metadata = self._get_align_model()
        from whisperx import align, load_audio

        audio = load_audio(str(audio_path))
        return align(
            segments,
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )["segments"]

    def _transcribe_segments(self, audio_path: Path, backend: str):
        backend = (backend or "whisperx").lower()
        if backend == "whisper":
            model, suppress_tokens = self._get_whisper_model()
            segments = model.transcribe(
                str(audio_path), suppress_tokens=suppress_tokens, word_timestamps=True
            )["segments"]
            for segment in segments:
                segment["text"] = self._replace_numbers_with_words(segment.get("text", ""))
            return segments

        whisperx_model = self._get_whisperx_model()
        segments = whisperx_model.transcribe(str(audio_path), batch_size=8)["segments"]
        for segment in segments:
            segment["text"] = self._replace_numbers_with_words(segment.get("text", ""))

        align_model, align_metadata = self._get_align_model()
        from whisperx import align, load_audio

        audio = load_audio(str(audio_path))
        return align(
            segments,
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )["segments"]

    def _build_target_transcript(
        self, segments, text: str, prompt_end_sec: float
    ) -> tuple[str, float]:
        words_info = [word for segment in segments for word in segment.get("words", [])]
        target_transcript = ""

        for word in words_info:
            if word.get("end", 0) < prompt_end_sec:
                target_transcript += word.get("word", "")
                if target_transcript and not target_transcript.endswith(" "):
                    target_transcript += " "
            elif (word.get("start", 0) + word.get("end", 0)) / 2 < prompt_end_sec:
                target_transcript += word.get("word", "")
                if target_transcript and not target_transcript.endswith(" "):
                    target_transcript += " "
                prompt_end_sec = word.get("end", prompt_end_sec)
                break
            else:
                break

        target_transcript = f"{target_transcript.strip()} {text}".strip()
        target_transcript = self._replace_numbers_with_words(target_transcript)
        return target_transcript, prompt_end_sec

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        import torch
        import torchaudio
        from inference_tts_scale import inference_one_sample

        ref_text = kwargs.get("text_reference") or kwargs.get("reference_text") or ""
        ref_text = str(ref_text).strip()
        smart_transcript = kwargs.get("smart_transcript", True)
        whisper_backend = kwargs.get("whisper_backend", "whisperx")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            temp_audio = tmpdir_path / "reference.wav"

            audio = torch.tensor(reference_audio, dtype=torch.float32)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if reference_sample_rate != self.SAMPLE_RATE:
                audio = torchaudio.functional.resample(
                    audio, reference_sample_rate, self.SAMPLE_RATE
                )
            sf.write(temp_audio, audio.squeeze(0).cpu().numpy(), self.SAMPLE_RATE)

            info = torchaudio.info(str(temp_audio))
            audio_dur = info.num_frames / info.sample_rate

            prompt_end_sec = kwargs.get("prompt_end_sec") or kwargs.get("cut_off_sec")
            try:
                prompt_end_sec = float(prompt_end_sec) if prompt_end_sec is not None else None
            except (TypeError, ValueError):
                prompt_end_sec = None

            if prompt_end_sec is None or prompt_end_sec <= 0:
                prompt_end_sec = self.cut_off_sec

            prompt_end_sec = min(prompt_end_sec, audio_dur)

            segments = None
            if smart_transcript:
                if ref_text:
                    segments = self._align_transcript(ref_text, temp_audio)
                else:
                    segments = self._transcribe_segments(temp_audio, whisper_backend)

            if smart_transcript and segments:
                target_transcript, prompt_end_sec = self._build_target_transcript(
                    segments, str(text), prompt_end_sec
                )
            else:
                ref_text_clean = self._replace_numbers_with_words(ref_text) if ref_text else ""
                target_parts = [
                    part.strip()
                    for part in [ref_text_clean, str(text)]
                    if part and str(part).strip()
                ]
                target_transcript = " ".join(target_parts)

            prompt_end_sec = min(prompt_end_sec, audio_dur)
            prompt_end_frame = int(prompt_end_sec * info.sample_rate)

            decode_config = {
                "top_k": self.top_k,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "stop_repetition": self.stop_repetition,
                "kvcache": self.kvcache,
                "codec_audio_sr": self.codec_audio_sr,
                "codec_sr": self.codec_sr,
                "silence_tokens": self.silence_tokens,
                "sample_batch_size": self.sample_batch_size,
            }

            _, gen_audio = inference_one_sample(
                self.model,
                self.model.args,
                self.model.args.phn2num,
                self.text_tokenizer,
                self.audio_tokenizer,
                str(temp_audio),
                target_transcript,
                self.device,
                decode_config,
                prompt_end_frame,
            )

            output_audio = gen_audio.squeeze()
            if isinstance(output_audio, torch.Tensor):
                output_audio = output_audio.detach().cpu().numpy()

        return output_audio.astype(np.float32), self.SAMPLE_RATE
