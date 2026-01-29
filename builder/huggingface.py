#!/usr/bin/env python3
"""Generate HuggingFace model card and upload weights.

Workflow:
  1. `prepare` - Downloads weights to models/<model>/weights/ (or runs model-specific script)
  2. `readme`  - Generates README.md in models/<model>/weights/
  3. `upload`  - Uploads the local weights/ directory to HuggingFace

The weights/ directory is git-ignored and stores the full repo locally for testing.
Models can have custom scripts at scripts/prepare_weights.py for special handling.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

try:
    # When run as a module: python -m builder.huggingface ...
    from .languages import (  # type: ignore
        get_language_choices_for_gradio,
        get_language_name,
    )
    from .names import normalize_name  # type: ignore
except Exception:
    # When run as a script: python builder/huggingface.py ...
    from languages import (  # type: ignore
        get_language_choices_for_gradio,
        get_language_name,
    )
    from names import normalize_name  # type: ignore


# License ID to human-readable name mapping
LICENSE_NAMES = {
    "mit": "MIT License",
    "apache-2.0": "Apache License 2.0",
    "gpl-3.0": "GNU General Public License v3.0",
    "cc-by-4.0": "Creative Commons Attribution 4.0",
    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
    "cc0-1.0": "Creative Commons Zero v1.0 Universal",
    "bsd-3-clause": "BSD 3-Clause License",
    "other": "Other (see original repository)",
}


def load_config(model_dir: Path) -> dict:
    """Load config.yaml from model directory."""
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_pyproject_version(model_dir: Path) -> str:
    """Read [project].version from pyproject.toml in model_dir."""
    pyproject_path = model_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"
    try:
        import tomllib  # py>=3.11
    except Exception:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return str(data.get("project", {}).get("version", "0.0.0"))


def get_huggingface_dir(model_dir: Path) -> Path:
    """Get the local weights directory for a model."""
    return model_dir / "weights"


def generate_readme(model_dir: Path, output_dir: Path | None = None) -> Path:
    """Generate HuggingFace README.md from model config.
    
    Args:
        model_dir: Path to the model directory containing config.yaml.
        output_dir: Output directory for README.md. Defaults to model_dir/weights/.
        
    Returns:
        Path to the generated README.md.
    """
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    code = config.get("code", {})
    weights = config.get("weights", {})
    external = config.get("external", {})
    api = config.get("api", {})
    api_params = api.get("parameters", {})
    base_model = config.get("weights", {}).get("url", [])

    if base_model and "huggingface.co" in base_model:
        base_model = base_model.replace("https://huggingface.co/", "")
    
    # Get name variants
    names = normalize_name(metadata.get("name", model_dir.name))
    
    # Set up output directory
    if output_dir is None:
        output_dir = get_huggingface_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up Jinja2
    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates" / "weights"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )
    
    # Get language codes and names
    language_codes = metadata.get("languages", [])
    language_names = [get_language_name(code) for code in language_codes]
    
    # Prepare template context
    context = {
        **names,
        "description": (metadata.get("description") or "").rstrip(),
        "weights_license": weights.get("license", "other"),
        "weights_license_name": LICENSE_NAMES.get(weights.get("license", "other"), "Other"),
        "code_license": code.get("license", "other"),
        "code_license_name": LICENSE_NAMES.get(code.get("license", "other"), "Other"),
        "weights_url": weights.get("url", ""),
        # Used for HF model card metadata (front matter). For these mirrors, the
        # upstream weights repo is the most useful `base_model`.
        "base_model": base_model or "",
        "code_url": code.get("url", ""),
        "language_codes": language_codes,
        "language_names": language_names,
        "architecture": metadata.get("architecture", []),
        "sample_rate": metadata.get("sample_rate", 22050),
        "num_parameters": metadata.get("num_parameters", "Unknown"),
        "release_date": metadata.get("release_date", "Unknown"),
        "training_data": metadata.get("training_data", []),
        "citations": external.get("citations", []),
        "paper_urls": external.get("paper_urls", []),
        "has_text_reference": "text_reference" in api_params,
        "has_language": "language" in api_params,
    }
    
    # Render README
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    template = env.get_template("README.md.j2")
    readme_content = template.render(
        **context,
        generated_at=generated_at,
        generated_from_template="templates/weights/README.md.j2",
    )
    
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"Generated {readme_path}")
    
    return readme_path


def download_weights(
    source_url: str,
    output_dir: Path,
    commit: str | None = None,
) -> Path:
    """Download weights from source URL.
    
    Supports HuggingFace repos (huggingface.co/*) via huggingface_hub.
    
    Args:
        source_url: URL to the weights (HuggingFace repo URL).
        output_dir: Directory to download weights to.
        commit: Optional commit/revision to download.
        
    Returns:
        Path to downloaded weights directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse HuggingFace URL
    if "huggingface.co" in source_url:
        # Extract repo_id from URL like https://huggingface.co/amphion/MaskGCT
        parts = source_url.rstrip("/").split("huggingface.co/")[-1]
        repo_id = parts.split("/tree/")[0]  # Handle URLs with branch/commit
        
        print(f"Downloading from HuggingFace: {repo_id}")
        
        from huggingface_hub import snapshot_download
        
        download_path = snapshot_download(
            repo_id=repo_id,
            revision=commit,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        return Path(download_path)
    else:
        raise ValueError(f"Unsupported weights URL format: {source_url}")


def prepare_weights(
    model_dir: Path,
    force: bool = False,
) -> Path:
    """Prepare weights for a model by downloading or running model-specific script.
    
    This downloads weights to the local weights/ directory for testing and upload.
    If a model-specific script exists at scripts/prepare_weights.py, it will be run instead.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        force: If True, re-download even if weights already exist.
        
    Returns:
        Path to the prepared weights directory.
    """
    model_dir = Path(model_dir).resolve()
    hf_dir = get_huggingface_dir(model_dir)
    
    # Check for model-specific prepare script
    custom_script = model_dir / "scripts" / "prepare_weights.py"
    if custom_script.exists():
        print(f"Running custom prepare script: {custom_script}")
        subprocess.run(
            [sys.executable, str(custom_script)],
            cwd=model_dir,
            check=True,
        )
        print("✓ Custom script completed")
        # Run generic post-processing (e.g., vendor asset extraction), then stop.
        _prepare_vendor_assets(model_dir, hf_dir, force=force)
        print("Generating README...")
        generate_readme(model_dir, hf_dir)
        return hf_dir
    
    # Check if weights already exist (look for actual model files, not just README)
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx"}
    has_weights = hf_dir.exists() and any(
        f.suffix in weight_extensions for f in hf_dir.rglob("*") if f.is_file()
    )
    if has_weights and not force:
        print(f"Weights already exist at {hf_dir}")
        print("Use --force to re-download")
        return hf_dir
    
    # Load config for default download
    config = load_config(model_dir)
    weights = config.get("weights", {})
    
    source_url = weights.get("url")
    commit = weights.get("commit")
    
    if not source_url:
        raise ValueError(
            f"No weights.url in config.yaml and no custom script at {custom_script}\n"
            "Either add weights.url to config.yaml or create a custom prepare script."
        )
    
    # Download weights
    print(f"Downloading weights from {source_url}...")
    download_weights(source_url, hf_dir, commit)
    
    # Extract any large vendor assets into the weights repo
    _prepare_vendor_assets(model_dir, hf_dir, force=force)

    # Generate README
    print("Generating README...")
    generate_readme(model_dir, hf_dir)
    
    print(f"✓ Weights prepared at {hf_dir}")
    return hf_dir


def _prepare_vendor_assets(model_dir: Path, hf_dir: Path, force: bool = False) -> None:
    """Copy configured vendor assets from upstream code repo into the HF weights folder.

    This is used when upstream research repositories include large/binary assets inside
    the code tree (e.g. ONNX models). We keep wheels lean by stripping these from vendored
    code and uploading them alongside model weights instead.
    Skipped when package.pypi is set (PyPI-only distribution, no vendored code).
    """
    config = load_config(model_dir)
    pypi_config = config.get("package", {}).get("pypi")
    if pypi_config:
        return
    code = config.get("code", {}) or {}
    vendor_assets = code.get("vendor_assets", []) or []
    if not vendor_assets:
        return

    url = code.get("url")
    commit = code.get("commit")
    if not url or not commit:
        raise ValueError("code.url and code.commit are required to fetch vendor_assets")

    tmp_dir = hf_dir / ".tmp_vendor_assets"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = tmp_dir / "repo"
    print(f"Fetching vendor assets from {url} @ {commit} ...")
    subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", commit], cwd=repo_dir, check=True)
    subprocess.run(["git", "checkout", commit], cwd=repo_dir, check=True)

    for entry in vendor_assets:
        src_rel = entry.get("source")
        if not src_rel:
            continue
        dst_rel = entry.get("dest") or f"vendor_assets/{src_rel}"

        src = repo_dir / src_rel
        dst = hf_dir / dst_rel
        if dst.exists() and not force:
            continue
        if not src.exists():
            raise FileNotFoundError(f"Vendor asset not found in upstream repo: {src_rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied vendor asset → {dst_rel}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def upload_weights(
    model_dir: Path,
    repo_id: str = "ttsds",
    dry_run: bool = False,
) -> str:
    """Upload model weights to HuggingFace from local weights/ directory.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        repo_id: Target HuggingFace org or repository. If just an org name (no /),
                 creates a repo named {org}/{model_id} for each model.
        dry_run: If True, print what would be done without uploading.
        
    Returns:
        URL to the uploaded model.
    """
    model_dir = Path(model_dir).resolve()
    hf_dir = get_huggingface_dir(model_dir)
    
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    
    model_id = metadata.get("id", model_dir.name)
    
    # If repo_id is just an org name (no /), create org/model_id repo
    if "/" not in repo_id:
        repo_id = f"{repo_id}/{model_id}"
    
    # Check that weights have been prepared (look for actual model files)
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx"}
    has_weights = hf_dir.exists() and any(
        f.suffix in weight_extensions for f in hf_dir.rglob("*") if f.is_file()
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No weight files found at {hf_dir}\n"
            "Run 'just hf-weights-prepare <model>' first to download/prepare weights."
        )
    
    if dry_run:
        print(f"Would upload weights for {model_id}")
        print(f"  Source: {hf_dir}")
        print(f"  Target: {repo_id}")
        return f"https://huggingface.co/{repo_id}"
    
    # Regenerate README before upload to ensure it's up to date
    print("Regenerating README...")
    generate_readme(model_dir, hf_dir)
    
    # Upload to HuggingFace
    print(f"Uploading to {repo_id}...")
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    api.upload_folder(
        folder_path=str(hf_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {model_id} weights",
    )
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"✓ Uploaded to {url}")
    return url


def get_space_dir(model_dir: Path) -> Path:
    """Get the local space directory for a model."""
    return model_dir / "space"


def generate_space(model_dir: Path, output_dir: Path | None = None) -> Path:
    """Generate HuggingFace Space files from model config.
    
    Args:
        model_dir: Path to the model directory containing config.yaml.
        output_dir: Output directory for space files. Defaults to model_dir/space/.
        
    Returns:
        Path to the generated space directory.
    """
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    code = config.get("code", {})
    weights = config.get("weights", {})
    external = config.get("external", {})
    dependencies = config.get("dependencies", {}) or {}
    package_version = _load_pyproject_version(model_dir)
    
    # Load test_data.yaml for examples
    test_data_path = model_dir / "test_data.yaml"
    test_data = {}
    if test_data_path.exists():
        with open(test_data_path) as f:
            test_data = yaml.safe_load(f) or {}
    
    # Get name variants
    names = normalize_name(metadata.get("name", model_dir.name))
    
    # Set up output directory
    if output_dir is None:
        output_dir = get_space_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up Jinja2
    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates" / "space"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )
    
    # Get language codes and names
    language_codes = metadata.get("languages", [])
    language_names = [get_language_name(code) for code in language_codes]
    language_choices = get_language_choices_for_gradio(language_codes)
    
    # Build examples from test_data (combining test sentences with reference audio)
    test_sentences = test_data.get("test_sentences", {})
    reference_audio_data = test_data.get("reference_audio", {})
    examples = []
    
    # Create examples directory for reference audio files
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    for lang_code, sentences in test_sentences.items():
        # Get reference audio for this language
        ref_data = reference_audio_data.get(lang_code, {})
        ref_url = ref_data.get("url", "")
        ref_text = ref_data.get("text", "")
        
        # Download reference audio if URL is provided
        ref_audio_path = ""
        if ref_url:
            # Determine file extension from URL
            ext = Path(ref_url).suffix or ".wav"
            ref_audio_filename = f"ref_{lang_code}{ext}"
            ref_audio_local = examples_dir / ref_audio_filename
            
            if not ref_audio_local.exists():
                try:
                    import urllib.request
                    print(f"Downloading reference audio for {lang_code}...")
                    urllib.request.urlretrieve(ref_url, ref_audio_local)
                    print(f"  -> {ref_audio_local}")
                except Exception as e:
                    print(f"Warning: Failed to download reference audio for {lang_code}: {e}")
            
            if ref_audio_local.exists():
                # Use relative path from space root
                ref_audio_path = f"examples/{ref_audio_filename}"
        
        for sentence in sentences:
            examples.append({
                "text": sentence.get("text", ""),
                "language": lang_code,
                "reference_audio": ref_audio_path,
                "reference_text": ref_text,
            })
    
    # System packages for apt (used by Space Dockerfile and Replicate cog.yaml)
    system_packages = dependencies.get("system_packages") or ["build-essential", "git"]

    # Get PyPI package name/version if specified (for PyPI-only packages)
    package_config = config.get("package", {})
    pypi_config = package_config.get("pypi")
    has_external_pypi = pypi_config and isinstance(pypi_config, dict)
    if has_external_pypi:
        pypi_package_name = pypi_config.get("name", names["import_name"])
        pypi_package_version = pypi_config.get("version", package_version)
    else:
        pypi_package_name = names["import_name"]
        pypi_package_version = package_version

    # Prepare template context
    context = {
        **names,
        "description": (metadata.get("description") or "").rstrip(),
        "python_venv": str(dependencies.get("python_venv") or dependencies.get("python") or "3.10"),
        "system_packages": system_packages,
        "hf_repo": (os.environ.get("TTSDB_HF_REPO") or "ttsds").strip(),
        "package_version": package_version,
        "pypi_package_name": pypi_package_name,
        "pypi_package_version": pypi_package_version,
        "has_external_pypi": has_external_pypi,
        "weights_license": weights.get("license", "other"),
        "weights_url": weights.get("url", ""),
        "code_url": code.get("url", ""),
        "language_codes": language_codes,
        "language_names": language_names,
        "language_choices": language_choices,
        "architecture": metadata.get("architecture", []),
        "sample_rate": metadata.get("sample_rate", 22050),
        "num_parameters": metadata.get("num_parameters", "Unknown"),
        "release_date": metadata.get("release_date", ""),
        "citations": external.get("citations", []),
        "paper_urls": external.get("paper_urls", []),
        "examples": examples,
    }
    
    # Generate all space files
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    for template_name in ["app.py.j2", "Dockerfile.j2", "README.md.j2"]:
        template = env.get_template(template_name)
        content = template.render(
            **context,
            generated_at=generated_at,
            generated_from_template=f"templates/space/{template_name}",
        )
        
        output_name = template_name.replace(".j2", "")
        output_path = output_dir / output_name
        output_path.write_text(content)
        print(f"Generated {output_path}")
    
    return output_dir


def _default_model_id_from_weights_url(url: str) -> str:
    """Derive HuggingFace repo_id from weights.url (e.g. amphion/MaskGCT)."""
    if not url or "huggingface.co/" not in url:
        return ""
    return url.rstrip("/").split("huggingface.co/")[-1].split("/tree/")[0].strip("/")


def generate_replicate(model_dir: Path, output_dir: Path | None = None) -> Path:
    """Generate Replicate/Cog files (cog.yaml, predict.py, requirements.txt).

    Uses the same config-driven system_packages and python version as the Space
    Dockerfile. predict.py is a thin wrapper around the ttsdb_<model> package.

    Args:
        model_dir: Path to the model directory containing config.yaml.
        output_dir: Output directory. Defaults to model_dir/replicate/.

    Returns:
        Path to the output directory (model_dir/replicate/ by default).
    """
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    dependencies = config.get("dependencies", {}) or {}
    weights = config.get("weights", {})
    package_version = _load_pyproject_version(model_dir)
    names = normalize_name(metadata.get("name", model_dir.name))
    system_packages = dependencies.get("system_packages") or ["build-essential", "git"]
    python_version = str(dependencies.get("python_venv") or dependencies.get("python") or "3.10")
    weights_url = weights.get("url", "")
    default_model_id = _default_model_id_from_weights_url(weights_url) or ""

    # Get PyPI package name/version if specified (for PyPI-only packages)
    package_config = config.get("package", {})
    pypi_config = package_config.get("pypi")
    has_external_pypi = pypi_config and isinstance(pypi_config, dict)
    if has_external_pypi:
        pypi_package_name = pypi_config.get("name", names["import_name"])
        pypi_package_version = pypi_config.get("version", package_version)
    else:
        pypi_package_name = names["import_name"]
        pypi_package_version = package_version

    if output_dir is None:
        output_dir = Path(model_dir).resolve() / "replicate"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates" / "replicate"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )

    context = {
        **names,
        "system_packages": system_packages,
        "python_version": python_version,
        "package_version": package_version,
        "pypi_package_name": pypi_package_name,
        "pypi_package_version": pypi_package_version,
        "has_external_pypi": has_external_pypi,
        "weights_url": weights_url,
        "default_model_id": default_model_id,
    }

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    for template_name, output_name in [
        ("cog.yaml.j2", "cog.yaml"),
        ("predict.py.j2", "predict.py"),
        ("requirements.txt.j2", "requirements.txt"),
    ]:
        template = env.get_template(template_name)
        content = template.render(
            **context,
            generated_at=generated_at,
            generated_from_template=f"templates/replicate/{template_name}",
        )
        output_path = output_dir / output_name
        output_path.write_text(content)
        print(f"Generated {output_path}")

    return output_dir


def upload_space(
    model_dir: Path,
    repo_id: str = "ttsds",
    dry_run: bool = False,
) -> str:
    """Upload space to HuggingFace Spaces.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        repo_id: Target HuggingFace org or space repo.
        dry_run: If True, print what would be done without uploading.
        
    Returns:
        URL to the uploaded space.
    """
    model_dir = Path(model_dir).resolve()
    space_dir = get_space_dir(model_dir)
    
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    
    model_id = metadata.get("id", model_dir.name)
    
    # If repo_id is just an org name (no /), create org/model_id-demo space
    if "/" not in repo_id:
        repo_id = f"{repo_id}/{model_id}"
    
    # Check that space has been generated
    if not space_dir.exists() or not (space_dir / "app.py").exists():
        raise FileNotFoundError(
            f"No space found at {space_dir}\n"
            "Run 'just hf-space-generate <model>' first to generate space files."
        )
    
    if dry_run:
        print(f"Would upload space for {model_id}")
        print(f"  Source: {space_dir}")
        print(f"  Target: {repo_id}")
        return f"https://huggingface.co/spaces/{repo_id}"
    
    # Upload to HuggingFace Spaces
    print(f"Uploading space to {repo_id}...")
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Create space repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    # Add HF_TOKEN as a space secret if available in environment or .env
    token = os.environ.get("HF_TOKEN")
    if not token:
        repo_root = Path(__file__).parent.parent
        env_path = repo_root / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if not line or line.strip().startswith("#"):
                    continue
                if line.strip().startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    if token:
        try:
            if hasattr(api, "add_space_secret"):
                func = getattr(api, "add_space_secret")
                # Try common invocation patterns to support different huggingface_hub versions
                try:
                    func(repo_id, "HF_TOKEN", token)
                except TypeError:
                    try:
                        func(repo_id=repo_id, name="HF_TOKEN", value=token)
                    except TypeError:
                        try:
                            func(space=repo_id, name="HF_TOKEN", value=token)
                        except Exception as e:
                            print(f"Warning: failed to add space secret (unexpected signature): {e}")
            else:
                print("Warning: huggingface_hub HfApi has no add_space_secret; skipping secret upload")
        except Exception as e:
            print(f"Warning: failed to add space secret: {e}")

    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Update {model_id} space",
    )
    
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"✓ Uploaded to {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace model card and upload weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Prepare weights locally:
     %(prog)s prepare models/maskgct
     
  2. (Optional) Regenerate README:
     %(prog)s readme models/maskgct
     
  3. Upload to HuggingFace:
     %(prog)s upload models/maskgct

    The weights are stored in models/<model>/weights/ which is git-ignored.
This allows local testing without re-downloading from HuggingFace.

Custom Scripts:
  Models can have a custom script at scripts/prepare_weights.py for special
  handling (e.g., downloading from multiple sources, converting formats).

Examples:
  # Prepare weights (download or run custom script)
  %(prog)s prepare models/maskgct
  
  # Force re-download
  %(prog)s prepare models/maskgct --force
  
  # Generate README only
  %(prog)s readme models/maskgct
  
  # Upload to ttsds/<model_id> (e.g., ttsds/maskgct)
  %(prog)s upload models/maskgct
  
  # Upload to custom org (creates myorg/<model_id>)
  %(prog)s upload models/maskgct --repo myorg
  
  # Upload to specific repo
  %(prog)s upload models/maskgct --repo myorg/custom-name
  
  # Dry run (show what would be done)
  %(prog)s upload models/maskgct --dry-run
  
  # Generate space files
  %(prog)s space models/maskgct
  
  # Upload space to HuggingFace
  %(prog)s space-upload models/maskgct
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # prepare subcommand
    prepare_parser = subparsers.add_parser(
        "prepare", 
        help="Download/prepare weights to local weights/ directory"
    )
    prepare_parser.add_argument("model_dir", type=Path, help="Model directory")
    prepare_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-download even if weights exist"
    )
    
    # readme subcommand
    readme_parser = subparsers.add_parser("readme", help="Generate HuggingFace README")
    readme_parser.add_argument("model_dir", type=Path, help="Model directory")
    readme_parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: <model_dir>/weights/)"
    )
    
    # upload subcommand
    upload_parser = subparsers.add_parser("upload", help="Upload weights to HuggingFace")
    upload_parser.add_argument("model_dir", type=Path, help="Model directory")
    upload_parser.add_argument(
        "--repo", "-r", default="ttsds",
        help="Target HuggingFace org or repo (default: ttsds → ttsds/<model_id>)"
    )
    upload_parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without uploading"
    )
    
    # space subcommand (generate space files)
    space_parser = subparsers.add_parser(
        "space", 
        help="Generate HuggingFace Space files (app.py, Dockerfile, README.md)"
    )
    space_parser.add_argument("model_dir", type=Path, help="Model directory")
    space_parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: <model_dir>/space/)"
    )
    
    # space-upload subcommand
    space_upload_parser = subparsers.add_parser(
        "space-upload", 
        help="Upload space to HuggingFace Spaces"
    )
    space_upload_parser.add_argument("model_dir", type=Path, help="Model directory")
    space_upload_parser.add_argument(
        "--repo", "-r", default="ttsds",
        help="Target HuggingFace org or space repo (default: ttsds → ttsds/<model_id>)"
    )
    space_upload_parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without uploading"
    )
    
    # replicate subcommand (generate Replicate/Cog files)
    replicate_parser = subparsers.add_parser(
        "replicate",
        help="Generate Replicate/Cog files (cog.yaml, predict.py, requirements.txt)",
    )
    replicate_parser.add_argument("model_dir", type=Path, help="Model directory")
    replicate_parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: <model_dir>/replicate/)",
    )
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_weights(args.model_dir, args.force)
    elif args.command == "readme":
        generate_readme(args.model_dir, args.output)
    elif args.command == "upload":
        upload_weights(args.model_dir, args.repo, args.dry_run)
    elif args.command == "space":
        generate_space(args.model_dir, args.output)
    elif args.command == "space-upload":
        upload_space(args.model_dir, args.repo, args.dry_run)
    elif args.command == "replicate":
        generate_replicate(args.model_dir, args.output)


if __name__ == "__main__":
    main()
