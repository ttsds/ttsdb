#!/usr/bin/env python3
"""Generate the repository root README from templates + model configs.

This script scans `models/*/config.yaml` and renders `README.md` from
`templates/root/README.md.j2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


@dataclass(frozen=True)
class SystemRow:
    name: str
    local_path: str
    training_data: str
    multilingual: str
    languages: str
    training_k_hours: str
    num_parameters_m: str
    target_repr: str
    nar: str
    ar: str
    diffusion: str


def _checkmark(v: bool) -> str:
    return "✅" if v else "❌"


def _fmt_k_hours(total_hours: float | int | None) -> str:
    if total_hours is None:
        return "Unknown"
    try:
        h = float(total_hours)
    except Exception:
        return "Unknown"
    if h <= 0:
        return "Unknown"
    k = h / 1000.0
    if abs(k - round(k)) < 1e-9:
        return str(int(round(k)))
    return f"{k:.2f}".rstrip("0").rstrip(".")


def _join_training_data(items: list[dict] | None) -> str:
    if not items:
        return "Unknown"
    names: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        n = (it.get("name") or it.get("id") or "").strip()
        if n:
            names.append(n)
    return ", ".join(names) if names else "Unknown"


def _sum_training_hours(items: list[dict] | None) -> float | None:
    if not items:
        return None
    total = 0.0
    saw_any = False
    for it in items:
        if not isinstance(it, dict):
            continue
        if "hours" not in it:
            continue
        saw_any = True
        try:
            total += float(it.get("hours") or 0)
        except Exception:
            # ignore unparsable hours entries
            pass
    return total if saw_any else None


def _normalize_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(v).strip()] if str(v).strip() else []


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def gather_system_rows(repo_root: Path) -> list[SystemRow]:
    models_dir = repo_root / "models"
    rows: list[SystemRow] = []

    for cfg_path in sorted(models_dir.glob("*/config.yaml")):
        model_dir = cfg_path.parent
        cfg = _load_config(cfg_path)
        meta = cfg.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        name = str(meta.get("name") or model_dir.name)
        local_path = f"models/{model_dir.name}"

        languages = _normalize_list(meta.get("languages"))
        multilingual = _checkmark(len(languages) > 1)
        languages_s = ", ".join(languages) if languages else "Unknown"

        training_items = meta.get("training_data") if isinstance(meta.get("training_data"), list) else []
        training_data = _join_training_data(training_items)  # type: ignore[arg-type]
        training_hours = _sum_training_hours(training_items)  # type: ignore[arg-type]
        training_k_hours = _fmt_k_hours(training_hours)

        num_parameters = meta.get("num_parameters")
        num_parameters_m = "Unknown" if num_parameters in (None, "") else str(num_parameters)

        target_repr_list = _normalize_list(meta.get("target_representation"))
        target_repr = ", ".join(target_repr_list) if target_repr_list else "Unknown"

        architecture = _normalize_list(meta.get("architecture"))
        arch_s = " ".join(architecture).lower()

        is_nar = "non-autoregressive" in arch_s
        # Avoid treating "Non-Autoregressive" as autoregressive due to substring match.
        is_ar = (("autoregressive" in arch_s) and not is_nar) or ("language modeling" in arch_s)

        nar = _checkmark(is_nar)
        ar = _checkmark(is_ar)
        diffusion = _checkmark(("diffusion" in arch_s) or ("flow matching" in arch_s))

        rows.append(
            SystemRow(
                name=name,
                local_path=local_path,
                training_data=training_data,
                multilingual=multilingual,
                languages=languages_s,
                training_k_hours=training_k_hours,
                num_parameters_m=num_parameters_m,
                target_repr=target_repr,
                nar=nar,
                ar=ar,
                diffusion=diffusion,
            )
        )

    return rows


def generate_root_readme(repo_root: Path) -> Path:
    templates_dir = repo_root / "templates" / "root"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), keep_trailing_newline=True)

    systems = gather_system_rows(repo_root)
    template = env.get_template("README.md.j2")
    content = template.render(systems=systems)

    out_path = repo_root / "README.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def main() -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    out = generate_root_readme(repo_root)
    print(f"Generated {out}")


if __name__ == "__main__":
    main()

