#!/usr/bin/env python3
"""Synthesize a dataset across models with GPU distribution and status tracking."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_DATASET = ASSETS_DIR / "test_data.yaml"


@dataclass
class Job:
    job_id: str
    model_id: str
    variant: str | None
    language: str
    text: str
    reference_audio: str
    text_reference: str
    output_relpath: str
    speaker: str | None = None


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _normalize_import_name(project_name: str) -> str:
    return project_name.replace("-", "_")


def _resolve_model_class_path(model_dir: Path) -> str:
    pyproject_path = model_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Missing pyproject.toml in {model_dir}")

    import tomllib

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    project_name = pyproject.get("project", {}).get("name")
    if not project_name:
        raise ValueError(f"Missing project.name in {pyproject_path}")

    import_name = _normalize_import_name(project_name)
    init_path = model_dir / "src" / import_name / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing {init_path}")

    class_name = None
    pattern = re.compile(r"class\s+(\w+)\(VoiceCloningTTSBase\)\s*:")
    for line in init_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            class_name = match.group(1)
            break
    if not class_name:
        raise ValueError(f"Could not find VoiceCloningTTSBase subclass in {init_path}")

    return f"{import_name}.{class_name}"


def _resolve_import_name(model_dir: Path) -> str:
    pyproject_path = model_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Missing pyproject.toml in {model_dir}")

    import tomllib

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    project_name = pyproject.get("project", {}).get("name")
    if not project_name:
        raise ValueError(f"Missing project.name in {pyproject_path}")
    return _normalize_import_name(project_name)


def _load_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {model_dir}")
    return _load_yaml(config_path)


def _iter_jobs(
    model_id: str,
    variant: str | None,
    config: dict,
    dataset: dict,
    output_dir: Path,
) -> list[Job]:
    metadata = config.get("metadata") or {}
    languages = metadata.get("languages") or []

    test_sentences = dataset.get("test_sentences") or {}
    reference_audio = dataset.get("reference_audio") or {}

    jobs: list[Job] = []
    variant_tag = variant or "default"

    for language in languages:
        sentences = test_sentences.get(language) or []
        ref = reference_audio.get(language)
        if not ref:
            continue
        ref_path = ref.get("path")
        ref_text = ref.get("text", "")
        if not ref_path:
            continue
        for idx, entry in enumerate(sentences):
            text = entry.get("text", "")
            if not text:
                continue
            job_id = f"{model_id}:{variant_tag}:{language}:{idx:03d}"
            output_rel = str(Path(model_id) / variant_tag / language / f"{idx:03d}.wav")
            jobs.append(
                Job(
                    job_id=job_id,
                    model_id=model_id,
                    variant=variant,
                    language=language,
                    text=text,
                    reference_audio=str(ROOT_DIR / ref_path),
                    text_reference=str(ref_text or ""),
                    output_relpath=output_rel,
                )
            )
    return jobs


def _find_suffix_file(paths: list[Path], suffix: str) -> Path | None:
    suffix = suffix.lower()
    for path in paths:
        stem = path.stem.lower()
        if stem.endswith(suffix) or stem.endswith(f"_{suffix}") or stem.endswith(f"-{suffix}"):
            return path
    return None


def _build_jobs_from_folder(
    model_ids: list[str],
    dataset_folder: Path,
    output_dir: Path,
    language: str,
    ref_suffix: str,
    target_suffix: str,
) -> list[Job]:
    jobs: list[Job] = []
    speakers = [p for p in dataset_folder.iterdir() if p.is_dir()]

    for model_id in model_ids:
        model_dir = MODELS_DIR / model_id
        config = _load_model_config(model_dir)
        variants = (config.get("variants") or {}).copy()
        default_variant = variants.pop("default", None)
        variant_list = [default_variant] if default_variant else [None]

        for variant in variant_list:
            variant_tag = variant or "default"
            for speaker_dir in speakers:
                audio_files = [
                    p
                    for p in speaker_dir.iterdir()
                    if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
                ]
                text_files = [p for p in speaker_dir.iterdir() if p.suffix.lower() in {".txt"}]

                ref_audio = _find_suffix_file(audio_files, ref_suffix)
                target_audio = _find_suffix_file(audio_files, target_suffix)
                ref_text = _find_suffix_file(text_files, ref_suffix)
                target_text = _find_suffix_file(text_files, target_suffix)

                if not (ref_audio and target_audio and ref_text and target_text):
                    print(
                        f"Skipping {speaker_dir.name}: missing ref/target files for suffixes "
                        f"{ref_suffix}/{target_suffix}"
                    )
                    continue

                text = target_text.read_text().strip()
                ref_text_content = ref_text.read_text().strip()
                job_id = f"{model_id}:{variant_tag}:{language}:{speaker_dir.name}"
                output_rel = str(
                    Path(model_id) / variant_tag / language / speaker_dir.name / "000.wav"
                )
                jobs.append(
                    Job(
                        job_id=job_id,
                        model_id=model_id,
                        variant=variant,
                        language=language,
                        text=text,
                        reference_audio=str(ref_audio),
                        text_reference=ref_text_content,
                        output_relpath=output_rel,
                        speaker=speaker_dir.name,
                    )
                )

    return jobs


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"jobs": {}, "summary": {"total": 0, "completed": 0, "failed": 0}}
    with path.open() as f:
        return json.load(f)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(status, f, indent=2)


def _update_summary(status: dict[str, Any]) -> None:
    jobs = status.get("jobs", {})
    total = len(jobs)
    completed = sum(1 for job in jobs.values() if job.get("status") == "completed")
    failed = sum(1 for job in jobs.values() if job.get("status") == "failed")
    status["summary"] = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "pending": max(0, total - completed - failed),
    }


def _build_jobs(
    model_ids: list[str],
    dataset_path: Path,
    output_dir: Path,
) -> list[Job]:
    dataset = _load_yaml(dataset_path)
    jobs: list[Job] = []

    if dataset.get("pairs"):
        return _build_jobs_from_pairs(model_ids, dataset, output_dir)

    for model_id in model_ids:
        model_dir = MODELS_DIR / model_id
        config = _load_model_config(model_dir)
        variants = (config.get("variants") or {}).copy()
        default_variant = variants.pop("default", None)
        variant_list = [default_variant] if default_variant else [None]

        for variant in variant_list:
            jobs.extend(_iter_jobs(model_id, variant, config, dataset, output_dir))

    return jobs


def _build_jobs_from_pairs(
    model_ids: list[str],
    dataset: dict,
    output_dir: Path,
) -> list[Job]:
    pairs = dataset.get("pairs") or []
    if not isinstance(pairs, list):
        raise ValueError("pairs must be a list")

    jobs: list[Job] = []
    for model_id in model_ids:
        model_dir = MODELS_DIR / model_id
        config = _load_model_config(model_dir)
        variants = (config.get("variants") or {}).copy()
        default_variant = variants.pop("default", None)
        variant_list = [default_variant] if default_variant else [None]

        for variant in variant_list:
            variant_tag = variant or "default"
            for idx, pair in enumerate(pairs):
                speaker = pair.get("speaker")
                language = pair.get("language", "eng")
                reference_audio = pair.get("reference_audio")
                reference_text = pair.get("reference_text", "")
                target_text = pair.get("target_text")

                if not reference_audio or not target_text:
                    continue

                speaker_tag = speaker or f"speaker_{idx:04d}"
                job_id = f"{model_id}:{variant_tag}:{language}:{speaker_tag}"
                output_rel = str(Path(model_id) / variant_tag / language / speaker_tag / "000.wav")
                jobs.append(
                    Job(
                        job_id=job_id,
                        model_id=model_id,
                        variant=variant,
                        language=language,
                        text=str(target_text),
                        reference_audio=str(reference_audio),
                        text_reference=str(reference_text or ""),
                        output_relpath=output_rel,
                        speaker=speaker,
                    )
                )

    return jobs


def _split_jobs(jobs: list[Job], gpu_ids: list[str]) -> dict[str, list[Job]]:
    shards: dict[str, list[Job]] = {gpu: [] for gpu in gpu_ids}
    for idx, job in enumerate(jobs):
        gpu = gpu_ids[idx % len(gpu_ids)]
        shards[gpu].append(job)
    return shards


def _ensure_model_ready(model_dir: Path) -> Path:
    import_name = _resolve_import_name(model_dir)
    vendor_dir = model_dir / "src" / import_name / "_vendor" / "source"
    if not vendor_dir.exists():
        fetch_cmd = ["just", "fetch", model_dir.name]
        subprocess.run(fetch_cmd, cwd=ROOT_DIR, check=True)

    venv_python = model_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        setup_cmd = ["just", "setup", model_dir.name, "gpu"]
        subprocess.run(setup_cmd, cwd=ROOT_DIR, check=True)

    if not venv_python.exists():
        raise FileNotFoundError(f"Missing venv for {model_dir.name} after setup.")

    return venv_python


def _start_worker(
    venv_python: Path,
    model_dir: Path,
    jobs: list[Job],
    output_dir: Path,
    gpu_id: str,
    whisper_model: str,
    no_wer: bool,
    device: str | None,
) -> dict[str, Any]:
    class_path = _resolve_model_class_path(model_dir)
    weights_path = model_dir / "weights"
    shard_dir = output_dir / ".shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    worker_script = ROOT_DIR / "builder" / "dataset_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"Missing dataset worker at {worker_script}")

    shard_path = shard_dir / f"{model_dir.name}_gpu{gpu_id}.json"
    result_path = shard_dir / f"{model_dir.name}_gpu{gpu_id}_results.json"

    shard_payload = [job.__dict__ for job in jobs]
    with shard_path.open("w") as f:
        json.dump(shard_payload, f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        str(venv_python),
        str(worker_script),
        "--model-class",
        class_path,
        "--model-path",
        str(weights_path),
        "--jobs",
        str(shard_path),
        "--output-dir",
        str(output_dir),
        "--result-file",
        str(result_path),
        "--whisper-model",
        whisper_model,
    ]
    if no_wer:
        cmd.append("--no-wer")
    if device:
        cmd.extend(["--device", device])

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return {
        "proc": proc,
        "jobs": jobs,
        "result_path": result_path,
        "gpu_id": gpu_id,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset)

    if args.all_models:
        model_ids = [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    else:
        model_ids = args.models or [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    gpu_ids = args.gpus.split(",") if args.gpus else ["0"]

    status_path = output_dir / "status.json"
    status = _load_status(status_path)
    if args.dataset_folder:
        dataset_folder = Path(args.dataset_folder)
        if not dataset_folder.exists():
            raise FileNotFoundError(dataset_folder)
        jobs = _build_jobs_from_folder(
            model_ids=model_ids,
            dataset_folder=dataset_folder,
            output_dir=output_dir,
            language=args.language,
            ref_suffix=args.ref_suffix,
            target_suffix=args.target_suffix,
        )
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        jobs = _build_jobs(model_ids, dataset_path, output_dir)

    existing_jobs = status.get("jobs", {})
    for job in jobs:
        if job.job_id not in existing_jobs:
            existing_jobs[job.job_id] = {
                "model_id": job.model_id,
                "variant": job.variant,
                "language": job.language,
                "speaker": job.speaker,
                "output_relpath": job.output_relpath,
                "status": "pending",
            }

    status["jobs"] = existing_jobs
    status["generated_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    _update_summary(status)
    _write_status(status_path, status)

    pending_jobs = [
        job for job in jobs if existing_jobs.get(job.job_id, {}).get("status") != "completed"
    ]

    if not pending_jobs:
        print("No pending jobs.")
        return 0

    jobs_by_model: dict[str, list[Job]] = defaultdict(list)
    for job in pending_jobs:
        jobs_by_model[job.model_id].append(job)

    for model_id, model_jobs in jobs_by_model.items():
        model_dir = MODELS_DIR / model_id
        venv_python = _ensure_model_ready(model_dir)
        shards = _split_jobs(model_jobs, gpu_ids)
        processes: list[dict[str, Any]] = []

        for gpu_id, shard_jobs in shards.items():
            if not shard_jobs:
                continue
            processes.append(
                _start_worker(
                    venv_python=venv_python,
                    model_dir=model_dir,
                    jobs=shard_jobs,
                    output_dir=output_dir,
                    gpu_id=gpu_id,
                    whisper_model=args.whisper_model,
                    no_wer=args.no_wer,
                    device=args.device,
                )
            )

        for entry in processes:
            proc = entry["proc"]
            stdout, stderr = proc.communicate()
            shard_jobs = entry["jobs"]
            result_path = entry["result_path"]

            if proc.returncode != 0:
                err_tail = (stderr or stdout or "").strip()
                if err_tail:
                    err_tail = "\n".join(err_tail.splitlines()[-20:])
                for job in shard_jobs:
                    existing_jobs[job.job_id] = {
                        "model_id": job.model_id,
                        "variant": job.variant,
                        "language": job.language,
                        "output_relpath": job.output_relpath,
                        "status": "failed",
                        "error": err_tail or "Worker failed with non-zero exit code.",
                    }
            else:
                with result_path.open() as f:
                    results = json.load(f)
                for result in results:
                    job_id = result.get("job_id")
                    if not job_id:
                        continue
                    job_status = existing_jobs.get(job_id, {})
                    job_status.update(result)
                    job_status["status"] = result.get(
                        "status", job_status.get("status", "completed")
                    )
                    existing_jobs[job_id] = job_status

            _update_summary(status)
            _write_status(status_path, status)

    print(json.dumps(status.get("summary", {}), indent=2))
    return 0


def show_status(args: argparse.Namespace) -> int:
    status_path = Path(args.output_dir) / "status.json"
    status = _load_status(status_path)
    _update_summary(status)
    summary = status.get("summary", {})
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize datasets across models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run synthesis")
    run_parser.add_argument("--models", nargs="*", default=None, help="Model ids to run")
    run_parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models (default variant only)",
    )
    run_parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="YAML dataset file")
    run_parser.add_argument(
        "--dataset-folder", default=None, help="Folder with per-speaker subdirectories"
    )
    run_parser.add_argument(
        "--language", default="eng", help="Language code for dataset-folder mode"
    )
    run_parser.add_argument("--ref-suffix", default="1", help="Suffix for reference audio/text")
    run_parser.add_argument("--target-suffix", default="2", help="Suffix for target audio/text")
    run_parser.add_argument(
        "--output-dir", default=str(ROOT_DIR / "outputs" / "dataset"), help="Output directory"
    )
    run_parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids")
    run_parser.add_argument("--whisper-model", default="base", help="Whisper model name")
    run_parser.add_argument("--no-wer", action="store_true", help="Disable WER computation")
    run_parser.add_argument("--device", default=None, help="Torch device override")
    run_parser.set_defaults(func=run)

    status_parser = subparsers.add_parser("status", help="Show progress")
    status_parser.add_argument(
        "--output-dir", default=str(ROOT_DIR / "outputs" / "dataset"), help="Output directory"
    )
    status_parser.set_defaults(func=show_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
