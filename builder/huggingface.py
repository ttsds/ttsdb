#!/usr/bin/env python3
"""Generate HuggingFace model card and upload weights.

Workflow:
  1. `prepare` - Downloads weights to models/<model>/huggingface/ (or runs model-specific script)
  2. `readme`  - Generates README.md in models/<model>/huggingface/
  3. `upload`  - Uploads the local huggingface/ directory to HuggingFace

The huggingface/ directory is git-ignored and stores the full repo locally for testing.
Models can have custom scripts at scripts/prepare_weights.py for special handling.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Environment, FileSystemLoader


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

# Language code to name mapping (ISO 639-1)
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    return LANGUAGE_NAMES.get(code, code)


def normalize_name(name: str) -> dict:
    """Generate all name variants from a model name."""
    model_name = name
    folder_name = name.lower().replace("_", "-").replace(" ", "-")
    folder_name = re.sub(r"-+", "-", folder_name)
    package_name = folder_name
    import_name = "ttsdb_" + folder_name.replace("-", "_")
    class_name = re.sub(r"[-\s]+", "", name)
    
    return {
        "model_name": model_name,
        "folder_name": folder_name,
        "package_name": package_name,
        "import_name": import_name,
        "class_name": class_name,
    }


def load_config(model_dir: Path) -> dict:
    """Load config.yaml from model directory."""
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_huggingface_dir(model_dir: Path) -> Path:
    """Get the local huggingface directory for a model."""
    return model_dir / "huggingface"


def generate_readme(model_dir: Path, output_dir: Optional[Path] = None) -> Path:
    """Generate HuggingFace README.md from model config.
    
    Args:
        model_dir: Path to the model directory containing config.yaml.
        output_dir: Output directory for README.md. Defaults to model_dir/huggingface/.
        
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
    
    # Get name variants
    names = normalize_name(metadata.get("name", model_dir.name))
    
    # Set up output directory
    if output_dir is None:
        output_dir = get_huggingface_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up Jinja2
    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates" / "huggingface"
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
        "weights_license": weights.get("license", "other"),
        "weights_license_name": LICENSE_NAMES.get(weights.get("license", "other"), "Other"),
        "code_license": code.get("license", "other"),
        "code_license_name": LICENSE_NAMES.get(code.get("license", "other"), "Other"),
        "weights_url": weights.get("url", ""),
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
    template = env.get_template("README.md.j2")
    readme_content = template.render(**context)
    
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"Generated {readme_path}")
    
    return readme_path


def download_weights(
    source_url: str,
    output_dir: Path,
    commit: Optional[str] = None,
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
    
    This downloads weights to the local huggingface/ directory for testing and upload.
    If a model-specific script exists at scripts/prepare_weights.py, it will be run instead.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        force: If True, re-download even if weights already exist.
        
    Returns:
        Path to the prepared huggingface directory.
    """
    model_dir = Path(model_dir).resolve()
    hf_dir = get_huggingface_dir(model_dir)
    
    # Check for model-specific prepare script
    custom_script = model_dir / "scripts" / "prepare_weights.py"
    if custom_script.exists():
        print(f"Running custom prepare script: {custom_script}")
        result = subprocess.run(
            [sys.executable, str(custom_script)],
            cwd=model_dir,
            check=True,
        )
        print(f"✓ Custom script completed")
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
    
    # Generate README
    print("Generating README...")
    generate_readme(model_dir, hf_dir)
    
    print(f"✓ Weights prepared at {hf_dir}")
    return hf_dir


def upload_weights(
    model_dir: Path,
    repo_id: str = "ttsds",
    dry_run: bool = False,
) -> str:
    """Upload model weights to HuggingFace from local huggingface/ directory.
    
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
            "Run 'just hf prepare <model>' first to download/prepare weights."
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


def publish_weights(
    model_dir: Path,
    repo_id: str = "ttsds",
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Prepare, generate README, and upload weights in one step.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        repo_id: Target HuggingFace org or repository.
        force: If True, re-download even if weights exist.
        dry_run: If True, print what would be done without uploading.
        
    Returns:
        URL to the uploaded model.
    """
    print("=" * 60)
    print("Step 1/3: Preparing weights...")
    print("=" * 60)
    prepare_weights(model_dir, force=force)
    
    print()
    print("=" * 60)
    print("Step 2/3: Generating README...")
    print("=" * 60)
    generate_readme(model_dir)
    
    print()
    print("=" * 60)
    print("Step 3/3: Uploading to HuggingFace...")
    print("=" * 60)
    return upload_weights(model_dir, repo_id, dry_run)


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

The weights are stored in models/<model>/huggingface/ which is git-ignored.
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
  
  # Do everything: prepare + readme + upload
  %(prog)s publish models/maskgct
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # prepare subcommand
    prepare_parser = subparsers.add_parser(
        "prepare", 
        help="Download/prepare weights to local huggingface/ directory"
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
        help="Output directory (default: <model_dir>/huggingface/)"
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
    
    # publish subcommand (prepare + readme + upload)
    publish_parser = subparsers.add_parser(
        "publish", 
        help="Prepare, generate README, and upload in one step"
    )
    publish_parser.add_argument("model_dir", type=Path, help="Model directory")
    publish_parser.add_argument(
        "--repo", "-r", default="ttsds",
        help="Target HuggingFace org or repo (default: ttsds → ttsds/<model_id>)"
    )
    publish_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-download even if weights exist"
    )
    publish_parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without uploading"
    )
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_weights(args.model_dir, args.force)
    elif args.command == "readme":
        generate_readme(args.model_dir, args.output)
    elif args.command == "upload":
        upload_weights(args.model_dir, args.repo, args.dry_run)
    elif args.command == "publish":
        publish_weights(args.model_dir, args.repo, args.force, args.dry_run)


if __name__ == "__main__":
    main()
