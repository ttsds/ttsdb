from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import VoiceCloningTTSBase


def _normalize_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_text(reference)
    hyp_words = _normalize_text(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[-1][-1] / max(1, len(ref_words))


def _resolve_model_class(model_class_path: str) -> type[VoiceCloningTTSBase]:
    module_name, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    if not issubclass(model_cls, VoiceCloningTTSBase):
        raise TypeError(f"{model_class_path} is not a VoiceCloningTTSBase subclass")
    return model_cls


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Jobs file must contain a list of job objects")
    return data


def _load_whisper(model_name: str | None):
    if not model_name:
        return None
    import whisper  # type: ignore

    return whisper.load_model(model_name)


def _transcribe(whisper_model, audio_path: Path, language: str | None) -> str:
    if whisper_model is None:
        return ""
    result = whisper_model.transcribe(
        str(audio_path),
        language=language or None,
    )
    return (result or {}).get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize dataset shard for a model")
    parser.add_argument("--model-class", required=True, help="Model class path")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--variant", default=None, help="Model variant")
    parser.add_argument("--jobs", required=True, help="JSON file with jobs list")
    parser.add_argument("--output-dir", required=True, help="Output directory root")
    parser.add_argument("--result-file", required=True, help="Path to write results JSON")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name")
    parser.add_argument("--no-wer", action="store_true", help="Disable WER computation")
    parser.add_argument("--device", default=None, help="Torch device override")

    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    output_dir = Path(args.output_dir)
    result_file = Path(args.result_file)

    jobs = _load_jobs(jobs_path)
    model_cls = _resolve_model_class(args.model_class)

    model = model_cls(model_path=str(args.model_path), variant=args.variant, device=args.device)

    whisper_model = None
    if not args.no_wer:
        whisper_model = _load_whisper(args.whisper_model)

    results: list[dict[str, Any]] = []

    for job in jobs:
        job_id = job.get("job_id")
        output_rel = job.get("output_relpath")
        language = job.get("language")
        text = job.get("text")
        reference_audio = job.get("reference_audio")

        result: dict[str, Any] = {
            "job_id": job_id,
            "output_relpath": output_rel,
            "language": language,
            "status": "completed",
        }

        try:
            audio, sr = model.synthesize(
                text=text,
                reference_audio=reference_audio,
                language=language,
                text_reference=job.get("text_reference", ""),
            )

            output_path = output_dir / output_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_audio(np.asarray(audio), int(sr), output_path)

            if whisper_model is not None:
                transcript = _transcribe(whisper_model, output_path, language)
                wer = _word_error_rate(text or "", transcript)
                result["transcript"] = transcript
                result["wer"] = wer
            results.append(result)
        except Exception as exc:  # pragma: no cover - best-effort failure handling
            result["status"] = "failed"
            result["error"] = str(exc)
            results.append(result)

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
