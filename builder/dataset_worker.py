from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ttsdb_core import VoiceCloningTTSBase


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize dataset shard for a model")
    parser.add_argument("--model-class", required=True, help="Model class path")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--variant", default=None, help="Model variant")
    parser.add_argument("--jobs", required=True, help="JSON file with jobs list")
    parser.add_argument("--output-dir", required=True, help="Output directory root")
    parser.add_argument("--result-file", required=True, help="Path to write results JSON")
    parser.add_argument("--device", default=None, help="Torch device override")

    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    output_dir = Path(args.output_dir)
    result_file = Path(args.result_file)

    jobs = _load_jobs(jobs_path)
    model_cls = _resolve_model_class(args.model_class)

    model = model_cls(model_path=str(args.model_path), variant=args.variant, device=args.device)

    results: list[dict[str, Any]] = []

    for job in jobs:
        job_id = job.get("job_id")
        output_rel = job.get("output_relpath")
        language = job.get("language")
        text = job.get("text")
        reference_audio = job.get("reference_audio")

        output_path = output_dir / output_rel

        result: dict[str, Any] = {
            "job_id": job_id,
            "output_relpath": output_rel,
            "language": language,
            "status": "completed",
        }

        try:
            if output_path.exists():
                result["skipped"] = "output_exists"
            else:
                audio, sr = model.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    language=language,
                    text_reference=job.get("text_reference", ""),
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_audio(np.asarray(audio), int(sr), output_path)
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
