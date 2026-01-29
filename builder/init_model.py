#!/usr/bin/env python3
"""Initialize a new TTS model from templates."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

try:
    # When run as a module: python -m builder.init_model ...
    from .names import normalize_name  # type: ignore
except Exception:
    # When run as a script: python builder/init_model.py ...
    from names import normalize_name  # type: ignore


def init_model(
    name: str,
    python_version: str = "3.10",
    torch_version: str = ">=2.0.0",
    python_requires: str | None = None,
    python_venv: str | None = None,
    hf_repo: str | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Initialize a new model from templates.
    
    Args:
        name: The model name.
        python_version: Python version for config.yaml.
        torch_version: Torch version constraint for config.yaml.
        output_dir: Output directory. Defaults to models/<folder_name>.
        dry_run: If True, print what would be created without creating files.
        
    Returns:
        Path to the created model directory.
    """
    # Python version policy:
    # - Use `python_venv` for the concrete interpreter used by `just setup`.
    # - Use `python_requires` for packaging metadata (PEP 440 specifier string).
    # - If only a single version is known to work, prefer pinning to that minor line:
    #     python_venv="3.11" and python_requires="==3.11.*"
    #
    # Backwards-compat:
    # - `python_version` may still be either a concrete interpreter ("3.11") OR
    #   a specifier (">=3.10,<3.12"). If `python_requires`/`python_venv` are not
    #   provided, we infer sensible defaults.
    python_arg = (python_version or "").strip() or "3.10"
    python_requires = (python_requires or "").strip() or None
    python_venv = (python_venv or "").strip() or None

    if python_requires is None and python_venv is None:
        is_spec = any(op in python_arg for op in ("<", ">", "=", "!", "~", ","))
        if is_spec:
            python_requires = python_arg
            m = re.search(r"(?:>=|==)\s*([0-9]+(?:\.[0-9]+)?)", python_arg)
            python_venv = m.group(1) if m else "3.10"
        else:
            python_venv = python_arg
            python_requires = f"=={python_venv}.*"
    else:
        # Fill missing pieces from the legacy python arg.
        if python_venv is None:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", python_arg)
            python_venv = m.group(1) if m else "3.10"
        if python_requires is None:
            python_requires = f"=={python_venv}.*"

    # Get name variants
    names = normalize_name(name)
    names["python_version"] = python_version  # kept for backwards compatibility in templates
    names["python_requires"] = python_requires
    names["python_venv"] = python_venv
    names["torch_version"] = torch_version
    hf_repo = (hf_repo or os.environ.get("TTSDB_HF_REPO") or "ttsds").strip()
    names["hf_repo"] = hf_repo
    names["hf_model_id"] = f"{hf_repo}/{names['folder_name']}"
    
    repo_root = Path(__file__).parent.parent

    # Set up paths
    templates_dir = repo_root / "templates" / "init"
    if output_dir is None:
        # Default: repo_root/models/<folder_name>
        output_dir = repo_root / "models" / names["folder_name"]
    
    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )
    
    # Template file mappings: (template_name, output_path)
    templates = [
        ("pyproject.toml.j2", "pyproject.toml"),
        ("config.yaml.j2", "config.yaml"),
        ("test_data.yaml.j2", "test_data.yaml"),
        ("README.md.j2", "README.md"),
        (".gitignore.j2", ".gitignore"),
        ("__init__.py.j2", f"src/{names['import_name']}/__init__.py"),
        ("vendor_init.py.j2", f"src/{names['import_name']}/_vendor/__init__.py"),
        ("conftest.py.j2", "tests/conftest.py"),
        ("test_model.py.j2", "tests/test_model.py"),
    ]
    
    if dry_run:
        print(f"Would create model '{names['model_name']}' at {output_dir}")
        print(f"  folder_name:     {names['folder_name']}")
        print(f"  package_name:    {names['package_name']}")
        print(f"  import_name:     {names['import_name']}")
        print(f"  class_name:      {names['class_name']}")
        print(f"  python_support:  {names['python_requires']}")
        print(f"  python_venv:     {names['python_venv']}")
        print(f"  hf_repo:         {names['hf_repo']}")
        print(f"  torch_version:   {names['torch_version']}")
        print("\nFiles that would be created:")
        for _, output_path in templates:
            print(f"  {output_dir / output_path}")
        return output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render templates
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    for template_name, output_path in templates:
        template = env.get_template(template_name)
        content = template.render(
            **names,
            generated_at=generated_at,
            generated_from_template=f"templates/init/{template_name}",
        )

        # After rendering config.yaml, parse it and feed key metadata back into
        # subsequent templates (README/test_data/etc).
        if template_name == "config.yaml.j2":
            try:
                cfg = yaml.safe_load(content) or {}
                meta = (cfg.get("metadata") or {}) if isinstance(cfg, dict) else {}
                names["description"] = (meta.get("description") or "").rstrip()
                langs = meta.get("languages") or []
                if isinstance(langs, str):
                    langs = [langs]
                names["language_codes"] = list(langs)
            except Exception:
                # Best-effort: templates still render even if parsing fails.
                names.setdefault("description", "")
                names.setdefault("language_codes", [])
        
        file_path = output_dir / output_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"Created {file_path}")
    
    print(f"\n✓ Model '{names['model_name']}' initialized at {output_dir}")
    print("\nNext steps:")
    print(f"  1. cd {output_dir}")
    print("  2. Fill in the TODO fields in config.yaml")
    print(f"  3. Implement _load_model and _synthesize in src/{names['import_name']}/__init__.py")
    print("  4. uv sync")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new TTS model from templates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s MaskGCT
  %(prog)s "XTTS v2" --python 3.11
  %(prog)s ParlerTTS --torch ">=2.1.0"
  %(prog)s MyModel --output ./custom/path
        """,
    )
    parser.add_argument(
        "name",
        help="Model name (e.g., 'MaskGCT', 'XTTS_v2', 'ParlerTTS')",
    )
    parser.add_argument(
        "--python", "-p",
        default="3.10",
        help="Legacy: either a venv interpreter (e.g. 3.11) or a specifier (e.g. >=3.10,<3.12).",
    )
    parser.add_argument(
        "--python-venv",
        default="",
        help="Concrete interpreter for venv creation (e.g. 3.11).",
    )
    parser.add_argument(
        "--python-requires",
        default="",
        help="PEP 440 specifier for supported Python versions (e.g. ==3.11.* or >=3.10,<3.12).",
    )
    parser.add_argument(
        "--hf-repo",
        default="",
        help="HuggingFace org/repo prefix for generated model_id (default: env TTSDB_HF_REPO or 'ttsds').",
    )
    parser.add_argument(
        "--torch", "-t",
        default=">=2.0.0",
        help="Torch version constraint (default: >=2.0.0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: models/<folder_name>)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be created without creating files",
    )
    
    args = parser.parse_args()
    init_model(
        args.name,
        python_version=args.python,
        torch_version=args.torch,
        python_requires=args.python_requires,
        python_venv=args.python_venv,
        hf_repo=args.hf_repo,
        output_dir=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
