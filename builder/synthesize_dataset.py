#!/usr/bin/env python3
"""Simple dataset synthesis orchestrator using subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
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


def _resolve_model_class_path(model_dir: Path) -> str:
    import_name = _resolve_import_name(model_dir)
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
                ref_text = _find_suffix_file(text_files, ref_suffix)
                target_text = _find_suffix_file(text_files, target_suffix)

                if not (ref_audio and ref_text and target_text):
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


def _build_jobs_from_pairs(
    model_ids: list[str],
    dataset: dict,
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


def _build_jobs(
    model_ids: list[str],
    dataset_path: Path,
    dataset_folder: Path | None,
    language: str,
    ref_suffix: str,
    target_suffix: str,
) -> list[Job]:
    if dataset_folder is not None:
        return _build_jobs_from_folder(
            model_ids, dataset_folder, language, ref_suffix, target_suffix
        )

    dataset = _load_yaml(dataset_path)
    if dataset.get("pairs"):
        return _build_jobs_from_pairs(model_ids, dataset)

    jobs: list[Job] = []
    for model_id in model_ids:
        model_dir = MODELS_DIR / model_id
        config = _load_model_config(model_dir)
        variants = (config.get("variants") or {}).copy()
        default_variant = variants.pop("default", None)
        variant_list = [default_variant] if default_variant else [None]

        for variant in variant_list:
            jobs.extend(_iter_jobs(model_id, variant, config, dataset))

    return jobs


def _split_jobs(jobs: list[Job], gpu_ids: list[str]) -> dict[str, list[Job]]:
    shards: dict[str, list[Job]] = {gpu: [] for gpu in gpu_ids}
    for idx, job in enumerate(jobs):
        gpu = gpu_ids[idx % len(gpu_ids)]
        shards[gpu].append(job)
    return shards


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


def _ensure_model_ready(model_dir: Path, recreate_venv: bool) -> Path:
    import_name = _resolve_import_name(model_dir)
    vendor_dir = model_dir / "src" / import_name / "_vendor" / "source"
    if not vendor_dir.exists():
        subprocess.run(["just", "fetch", model_dir.name], cwd=ROOT_DIR, check=True)

    venv_python = model_dir / ".venv" / "bin" / "python"
    if recreate_venv or not venv_python.exists():
        subprocess.run(["just", "setup", model_dir.name, "gpu"], cwd=ROOT_DIR, check=True)

    if not venv_python.exists():
        raise FileNotFoundError(f"Missing venv for {model_dir.name} after setup.")

    return venv_python


def _start_worker(
    venv_python: Path,
    model_dir: Path,
    jobs: list[Job],
    output_dir: Path,
    gpu_id: str,
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
    ]
    variant = jobs[0].variant if jobs else None
    if variant:
        cmd.extend(["--variant", variant])
    if device:
        cmd.extend(["--device", device])

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return {
        "proc": proc,
        "model_id": model_dir.name,
        "jobs": jobs,
        "result_path": result_path,
        "gpu_id": gpu_id,
    }


def _stream_worker_output(proc: subprocess.Popen[str], label: str) -> list[str]:
    lines: list[str] = []
    if proc.stdout is None:
        return lines
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(f"[{label}] {line}")
        lines.append(line)
        if len(lines) > 200:
            lines = lines[-200:]
    return lines


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
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(1, len(ref_words))


def _ensure_whisper(model_name: str):
    try:
        import whisper  # type: ignore

        return whisper.load_model(model_name, device="cpu")
    except Exception as exc:
        try:
            subprocess.run(
                [sys.executable, "-c", "import pip"],
                cwd=ROOT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                cwd=ROOT_DIR,
                check=True,
            )

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "openai-whisper"],
                cwd=ROOT_DIR,
                check=True,
            )
            import whisper  # type: ignore

            return whisper.load_model(model_name, device="cpu")
        except Exception as install_exc:
            print(
                "[wer] Whisper unavailable; skipping WER. "
                f"Install error: {install_exc} (initial: {exc})"
            )
            return None


def _compute_wer(
    jobs: list[Job],
    status: dict[str, Any],
    output_dir: Path,
    model_name: str,
) -> None:
    whisper_model = _ensure_whisper(model_name)
    if whisper_model is None:
        return
    jobs_by_id = {job.job_id: job for job in jobs}

    lang_map = {
        "eng": "en",
        "en": "en",
        "zho": "zh",
        "zh": "zh",
        "jpn": "ja",
        "ja": "ja",
        "kor": "ko",
        "ko": "ko",
        "yue": "yue",
        "deu": "de",
        "fra": "fr",
        "ita": "it",
        "spa": "es",
        "por": "pt",
        "nld": "nl",
        "pol": "pl",
        "rus": "ru",
        "ara": "ar",
        "ces": "cs",
        "hun": "hu",
        "hin": "hi",
        "tur": "tr",
    }

    for job_id, job_status in status.get("jobs", {}).items():
        if job_status.get("status") != "completed":
            continue
        if job_status.get("wer") is not None:
            continue
        job = jobs_by_id.get(job_id)
        if not job:
            continue
        output_path = output_dir / job.output_relpath
        if not output_path.exists():
            continue

        language = lang_map.get((job.language or "").lower())
        result = whisper_model.transcribe(
            str(output_path),
            language=language,
            fp16=False,
        )
        transcript = (result or {}).get("text", "").strip()
        wer = _word_error_rate(job.text or "", transcript)
        job_status["transcript"] = transcript
        job_status["wer"] = wer


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset)
    dataset_folder = Path(args.dataset_folder) if args.dataset_folder else None

    if args.all_models:
        model_ids = [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    else:
        model_ids = args.models or [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    gpu_ids = args.gpus.split(",") if args.gpus else ["0"]

    if dataset_folder is None and not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    status_path = output_dir / "status.json"
    status = _load_status(status_path)
    jobs = _build_jobs(
        model_ids=model_ids,
        dataset_path=dataset_path,
        dataset_folder=dataset_folder,
        language=args.language,
        ref_suffix=args.ref_suffix,
        target_suffix=args.target_suffix,
    )

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
        else:
            existing_jobs[job.job_id].setdefault("output_relpath", job.output_relpath)

    status["jobs"] = existing_jobs
    status["generated_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    for job in jobs:
        output_path = output_dir / job.output_relpath
        job_status = existing_jobs.get(job.job_id, {})
        if output_path.exists() and job_status.get("status") != "completed":
            job_status["status"] = "completed"
            job_status["skipped"] = "output_exists"
            existing_jobs[job.job_id] = job_status

    _update_summary(status)
    _write_status(status_path, status)

    pending_jobs = [
        job for job in jobs if existing_jobs.get(job.job_id, {}).get("status") != "completed"
    ]
    if pending_jobs:
        jobs_by_model: dict[str, list[Job]] = defaultdict(list)
        for job in pending_jobs:
            jobs_by_model[job.model_id].append(job)

        for model_id, model_jobs in jobs_by_model.items():
            model_dir = MODELS_DIR / model_id
            model_output_dir = output_dir / model_id
            expected_outputs = [output_dir / job.output_relpath for job in model_jobs]
            if expected_outputs and all(path.exists() for path in expected_outputs):
                print(f"Skipping {model_id}: all outputs already exist in {model_output_dir}")
                continue
            venv_python = _ensure_model_ready(model_dir, args.recreate_venv)
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
                        device=args.device,
                    )
                )

            for entry in processes:
                proc = entry["proc"]
                label = f"{entry['model_id']}:gpu{entry['gpu_id']}"
                output_lines = _stream_worker_output(proc, label)
                proc.wait()
                shard_jobs = entry["jobs"]
                result_path = entry["result_path"]

                if proc.returncode != 0:
                    err_tail = "\n".join(output_lines[-20:]) if output_lines else ""
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

    if not args.no_wer:
        _compute_wer(
            jobs=jobs,
            status=status,
            output_dir=output_dir,
            model_name=args.whisper_model,
        )
        _update_summary(status)
        _write_status(status_path, status)

    print(json.dumps(status.get("summary", {}), indent=2))
    return 0


def show_status(args: argparse.Namespace) -> int:
    status_path = Path(args.output_dir) / "status.json"
    try:
        from rich.align import Align
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
    except Exception:
        status = _load_status(status_path)
        _update_summary(status)
        summary = status.get("summary", {})
        print(json.dumps(summary, indent=2))
        return 0

    console = Console()
    output_dir = Path(args.output_dir)

    def _apply_output_exists(status: dict[str, Any]) -> bool:
        changed = False
        for job in status.get("jobs", {}).values():
            output_relpath = job.get("output_relpath")
            if not output_relpath:
                continue
            output_path = output_dir / output_relpath
            if output_path.exists() and job.get("status") != "completed":
                job["status"] = "completed"
                job["skipped"] = "output_exists"
                changed = True
        return changed

    def _compute_model_stats(status: dict[str, Any]) -> list[dict[str, Any]]:
        stats: dict[str, dict[str, int]] = {}
        for job in status.get("jobs", {}).values():
            model_id = job.get("model_id") or "unknown"
            model = stats.setdefault(
                model_id, {"total": 0, "completed": 0, "failed": 0, "pending": 0}
            )
            model["total"] += 1
            state = job.get("status") or "pending"
            if state == "completed":
                model["completed"] += 1
            elif state == "failed":
                model["failed"] += 1
            else:
                model["pending"] += 1
        rows = []
        for model_id, model in stats.items():
            total = model["total"]
            completed = model["completed"]
            failed = model["failed"]
            pending = model["pending"]
            percent = (completed / total * 100.0) if total else 0.0
            rows.append(
                {
                    "model_id": model_id,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "pending": pending,
                    "percent": percent,
                }
            )
        return sorted(rows, key=lambda item: item["model_id"])

    def _compute_gpu_stats() -> list[dict[str, Any]]:
        shard_dir = output_dir / ".shards"
        if not shard_dir.exists():
            return []
        totals: dict[str, int] = {}
        done: dict[str, int] = {}
        completed: dict[str, int] = {}
        failed: dict[str, int] = {}

        for shard_path in shard_dir.glob("*_gpu*.json"):
            name = shard_path.stem
            if name.endswith("_results"):
                continue
            parts = name.rsplit("_gpu", 1)
            if len(parts) != 2:
                continue
            gpu_id = parts[1]
            try:
                with shard_path.open() as f:
                    shard_jobs = json.load(f)
            except Exception:
                shard_jobs = []
            totals[gpu_id] = totals.get(gpu_id, 0) + len(shard_jobs)

            results_path = shard_path.with_name(f"{shard_path.stem}_results.json")
            if results_path.exists():
                try:
                    with results_path.open() as f:
                        results = json.load(f)
                except Exception:
                    results = []
                done[gpu_id] = done.get(gpu_id, 0) + len(results)
                completed[gpu_id] = completed.get(gpu_id, 0) + sum(
                    1 for r in results if r.get("status") == "completed"
                )
                failed[gpu_id] = failed.get(gpu_id, 0) + sum(
                    1 for r in results if r.get("status") == "failed"
                )

        rows = []
        for gpu_id in sorted(totals, key=lambda item: int(item) if item.isdigit() else item):
            rows.append(
                {
                    "gpu_id": gpu_id,
                    "total": totals.get(gpu_id, 0),
                    "done": done.get(gpu_id, 0),
                    "completed": completed.get(gpu_id, 0),
                    "failed": failed.get(gpu_id, 0),
                }
            )
        return rows

    def _render() -> Group:
        status = _load_status(status_path)
        changed = _apply_output_exists(status)
        _update_summary(status)
        summary = status.get("summary", {})
        if changed and args.reconcile:
            _write_status(status_path, status)

        status_mtime = None
        if status_path.exists():
            status_mtime = datetime.fromtimestamp(status_path.stat().st_mtime, timezone.utc)

        summary_table = Table(title="Overall")
        summary_table.add_column("Total", justify="right")
        summary_table.add_column("Completed", justify="right")
        summary_table.add_column("Failed", justify="right")
        summary_table.add_column("Pending", justify="right")
        summary_table.add_row(
            str(summary.get("total", 0)),
            str(summary.get("completed", 0)),
            str(summary.get("failed", 0)),
            str(summary.get("pending", 0)),
        )
        if status_mtime:
            summary_table.caption = (
                f"Output dir: {output_dir} | status.json updated: {status_mtime.isoformat()}"
            )

        overall_done = summary.get("completed", 0) + summary.get("failed", 0)
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        )
        progress.add_task("Overall progress", total=summary.get("total", 0), completed=overall_done)

        gpu_stats = _compute_gpu_stats()
        for gpu in gpu_stats:
            progress.add_task(
                f"GPU {gpu['gpu_id']}",
                total=gpu["total"],
                completed=gpu["done"],
            )

        model_table = Table(title="Per-model")
        model_table.add_column("Model")
        model_table.add_column("Completed", justify="right")
        model_table.add_column("Failed", justify="right")
        model_table.add_column("Pending", justify="right")
        model_table.add_column("Total", justify="right")
        model_table.add_column("Done%", justify="right")

        for row in _compute_model_stats(status):
            model_table.add_row(
                row["model_id"],
                str(row["completed"]),
                str(row["failed"]),
                str(row["pending"]),
                str(row["total"]),
                f"{row['percent']:.1f}%",
            )

        return Group(
            Panel(summary_table),
            Panel(progress, title="Progress"),
            Panel(Align.left(model_table)),
        )

    if args.watch:
        refresh_rate = max(0.1, 1.0 / max(args.interval, 0.1))
        with Live(_render(), console=console, refresh_per_second=refresh_rate) as live:
            while True:
                time.sleep(args.interval)
                live.update(_render())
    else:
        console.print(_render())
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
    run_parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Recreate model venvs before running",
    )
    run_parser.add_argument("--whisper-model", default="base", help="Whisper model name")
    run_parser.add_argument("--no-wer", action="store_true", help="Disable WER computation")
    run_parser.add_argument("--device", default=None, help="Torch device override")
    run_parser.set_defaults(func=run)

    status_parser = subparsers.add_parser("status", help="Show progress")
    status_parser.add_argument(
        "--output-dir", default=str(ROOT_DIR / "outputs" / "dataset"), help="Output directory"
    )
    status_parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh status continuously",
    )
    status_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds when using --watch",
    )
    status_parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Mark outputs that exist on disk as completed in status.json",
    )
    status_parser.set_defaults(func=show_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
