#!/usr/bin/env python3
"""Aggregate per-model integration_result.json into status/ and update README badges."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _collect_model_results(repo_root: Path) -> tuple[list[dict], bool, bool]:
    """Collect integration_result.json from each model dir.

    Returns (models list, overall_passed, any_run).
    overall_passed: True if no run failed; False if any run failed.
    any_run: True if at least one model has integration_result.json.
    """
    models_dir = repo_root / "models"
    if not models_dir.exists():
        return [], True, False

    model_dirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
    entries: list[dict] = []
    all_passed = True
    any_run = False

    for model_dir in model_dirs:
        result_path = model_dir / "integration_result.json"
        data = _load_result(result_path)
        if data is not None:
            entries.append(data)
            any_run = True
            if not data.get("passed", False):
                all_passed = False
        else:
            entries.append(
                {
                    "model_id": model_dir.name,
                    "passed": None,
                    "status": "not run",
                    "timestamp": None,
                    "variants_tested": [],
                    "artifacts_count": 0,
                }
            )

    return entries, all_passed, any_run


def _write_integration_status_json(
    repo_root: Path, models: list[dict], overall_passed: bool, any_run: bool
) -> Path:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "generated_at": generated_at,
        "overall_passed": overall_passed,
        "any_run": any_run,
        "models": models,
    }
    status_dir = repo_root / "status"
    status_dir.mkdir(exist_ok=True)
    out_path = status_dir / "integration_status.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _write_status_readme(
    repo_root: Path, models: list[dict], overall_passed: bool, any_run: bool
) -> Path:
    status_dir = repo_root / "status"
    status_dir.mkdir(exist_ok=True)
    out_path = status_dir / "README.md"

    rows = []
    for m in models:
        model_id = m.get("model_id", "?")
        passed = m.get("passed")
        if passed is True:
            status = "passing"
        elif passed is False:
            status = "failing"
        else:
            status = "not run"
        ts = m.get("timestamp") or "—"
        variants = ", ".join(m.get("variants_tested") or []) or "—"
        count = m.get("artifacts_count", 0)
        rows.append(
            f"| [{model_id}](../models/{model_id}) | {status} | {ts} | {variants} | {count} |"
        )

    overall = "passing" if (any_run and overall_passed) else ("failing" if any_run else "not run")
    table = "\n".join(
        [
            "| Model | Status | Last run | Variants | Artifacts |",
            "|-------|--------|----------|----------|-----------|",
            *rows,
        ]
    )
    body = f"""# Integration status

Overall: **{overall}**

{table}

Generated from `models/*/integration_result.json`. Update by running integration tests and then `just status-integration` (or `just test-integration-all`).
"""
    out_path.write_text(body)
    return out_path


def _badge_section(models: list[dict], overall_passed: bool, any_run: bool) -> str:
    """Generate markdown for README badge section."""
    lines = [
        "<!-- BEGIN BADGES -->",
        "[![ttsdb-core](https://img.shields.io/pypi/v/ttsdb-core)](https://pypi.org/project/ttsdb-core/)",
    ]
    if any_run and overall_passed:
        overall_label, overall_color = "passing", "green"
    elif any_run:
        overall_label, overall_color = "failing", "red"
    else:
        overall_label, overall_color = "not%20run", "lightgrey"
    lines.append(
        f"[![integration](https://img.shields.io/badge/integration-{overall_label}-{overall_color})](status/README.md)"
    )
    for m in models:
        model_id = m.get("model_id", "?")
        passed = m.get("passed")
        if passed is True:
            label, color = "passing", "green"
        elif passed is False:
            label, color = "failing", "red"
        else:
            label, color = "not%20run", "lightgrey"
        # Shield label: use model_id with hyphens allowed
        lines.append(
            f"[![{model_id}](https://img.shields.io/badge/{model_id.replace('-', '--')}-{label}-{color})](models/{model_id})"
        )
    lines.append("<!-- END BADGES -->")
    return "\n".join(lines) + "\n"


def _update_readme_badges(repo_root: Path, badge_md: str) -> bool:
    readme = repo_root / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text()
    pattern = re.compile(r"<!-- BEGIN BADGES -->.*?<!-- END BADGES -->", re.DOTALL)
    if not pattern.search(text):
        return False
    new_text = pattern.sub(badge_md.strip(), text)
    if new_text != text:
        readme.write_text(new_text)
        return True
    return False


def main() -> None:
    update_readme = "--readme" in sys.argv
    if update_readme:
        sys.argv.remove("--readme")

    repo_root = _repo_root()
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1]).resolve()

    models, overall_passed, any_run = _collect_model_results(repo_root)
    _write_integration_status_json(repo_root, models, overall_passed, any_run)
    _write_status_readme(repo_root, models, overall_passed, any_run)
    badge_md = _badge_section(models, overall_passed, any_run)
    if update_readme:
        updated = _update_readme_badges(repo_root, badge_md)
        if updated:
            print("Updated README.md badges")
        else:
            print("README.md has no <!-- BEGIN BADGES -->...<!-- END BADGES --> section; skipping")
    overall = "passing" if (any_run and overall_passed) else ("failing" if any_run else "not run")
    print(f"Wrote status/integration_status.json and status/README.md (overall: {overall})")


if __name__ == "__main__":
    main()
