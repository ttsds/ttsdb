#!/usr/bin/env python3
"""Generate HuggingFace model card and upload weights."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
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
        output_dir = model_dir / "huggingface"
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


def upload_weights(
    model_dir: Path,
    repo_id: str = "ttsds/models",
    subfolder: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """Upload model weights to HuggingFace.
    
    Args:
        model_dir: Path to model directory with config.yaml.
        repo_id: Target HuggingFace repository.
        subfolder: Subfolder within repo (defaults to model name).
        dry_run: If True, print what would be done without uploading.
        
    Returns:
        URL to the uploaded model.
    """
    config = load_config(model_dir)
    metadata = config.get("metadata", {})
    weights = config.get("weights", {})
    
    model_name = metadata.get("name", model_dir.name)
    if subfolder is None:
        subfolder = model_name
    
    source_url = weights.get("url")
    commit = weights.get("commit")
    
    if not source_url:
        raise ValueError("No weights.url specified in config.yaml")
    
    if dry_run:
        print(f"Would upload weights for {model_name}")
        print(f"  Source: {source_url} (commit: {commit})")
        print(f"  Target: {repo_id}/{subfolder}")
        return f"https://huggingface.co/{repo_id}/tree/main/{subfolder}"
    
    # Create temp directory for downloads
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        weights_dir = tmp_path / "weights"
        
        # Download weights
        print(f"Downloading weights from {source_url}...")
        download_weights(source_url, weights_dir, commit)
        
        # Generate README
        print("Generating README...")
        readme_path = generate_readme(model_dir, weights_dir)
        
        # Upload to HuggingFace
        print(f"Uploading to {repo_id}/{subfolder}...")
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.upload_folder(
            folder_path=str(weights_dir),
            repo_id=repo_id,
            path_in_repo=subfolder,
            repo_type="model",
            commit_message=f"Add {model_name} weights",
        )
    
    url = f"https://huggingface.co/{repo_id}/tree/main/{subfolder}"
    print(f"✓ Uploaded to {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace model card and upload weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate README only
  %(prog)s readme models/maskgct
  
  # Upload weights to ttsds/models
  %(prog)s upload models/maskgct
  
  # Upload to custom repo
  %(prog)s upload models/maskgct --repo myorg/myrepo
  
  # Dry run (show what would be done)
  %(prog)s upload models/maskgct --dry-run
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
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
        "--repo", "-r", default="ttsds/models",
        help="Target HuggingFace repo (default: ttsds/models)"
    )
    upload_parser.add_argument(
        "--subfolder", "-s", default=None,
        help="Subfolder in repo (default: model name)"
    )
    upload_parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without uploading"
    )
    
    args = parser.parse_args()
    
    if args.command == "readme":
        generate_readme(args.model_dir, args.output)
    elif args.command == "upload":
        upload_weights(args.model_dir, args.repo, args.subfolder, args.dry_run)


if __name__ == "__main__":
    main()
