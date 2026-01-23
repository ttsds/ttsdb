#!/usr/bin/env python3
"""Initialize a new TTS model from templates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader


def normalize_name(name: str) -> dict:
    """Generate all name variants from a model name.
    
    Args:
        name: The full model name (e.g., "MaskGCT", "XTTS_v2", "F5-TTS")
        
    Returns:
        Dictionary with all name variants:
        - model_name: Original name for display (e.g., "MaskGCT", "F5-TTS")
        - folder_name: Lowercase with hyphens (e.g., "maskgct", "f5-tts")
        - package_name: Same as folder_name (e.g., "maskgct")
        - import_name: Lowercase with underscores, prefixed (e.g., "ttsdb_maskgct")
        - class_name: Original with hyphens/spaces removed (e.g., "MaskGCT", "F5TTS")
    """
    # Original name for display
    model_name = name
    
    # Folder name: lowercase, underscores to hyphens
    folder_name = name.lower().replace("_", "-").replace(" ", "-")
    # Remove consecutive hyphens
    folder_name = re.sub(r"-+", "-", folder_name)
    
    # Package name: same as folder name (pip/uv style)
    package_name = folder_name
    
    # Import name: ttsdb_ prefix, underscores for Python imports
    import_name = "ttsdb_" + folder_name.replace("-", "_")
    
    # Class name: preserve original casing, only remove invalid Python identifier chars
    # Replace hyphens and spaces (invalid in identifiers), keep underscores
    class_name = re.sub(r"[-\s]+", "", name)
    
    return {
        "model_name": model_name,
        "folder_name": folder_name,
        "package_name": package_name,
        "import_name": import_name,
        "class_name": class_name,
    }


def init_model(
    name: str,
    python_version: str = "3.10",
    torch_version: str = ">=2.0.0",
    output_dir: Optional[Path] = None,
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
    # Get name variants
    names = normalize_name(name)
    names["python_version"] = python_version
    names["torch_version"] = torch_version
    
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
        ("README.md.j2", "README.md"),
        (".gitignore.j2", ".gitignore"),
        ("__init__.py.j2", f"src/{names['import_name']}/__init__.py"),
        ("conftest.py.j2", "tests/conftest.py"),
        ("test_model.py.j2", "tests/test_model.py"),
    ]
    
    if dry_run:
        print(f"Would create model '{names['model_name']}' at {output_dir}")
        print(f"  folder_name:     {names['folder_name']}")
        print(f"  package_name:    {names['package_name']}")
        print(f"  import_name:     {names['import_name']}")
        print(f"  class_name:      {names['class_name']}")
        print(f"  python_version:  {names['python_version']}")
        print(f"  torch_version:   {names['torch_version']}")
        print("\nFiles that would be created:")
        for template_name, output_path in templates:
            print(f"  {output_dir / output_path}")
        return output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render templates
    for template_name, output_path in templates:
        template = env.get_template(template_name)
        content = template.render(**names)
        
        file_path = output_dir / output_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"Created {file_path}")
    
    print(f"\n✓ Model '{names['model_name']}' initialized at {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. Fill in the TODO fields in config.yaml")
    print(f"  3. Implement _load_model and _synthesize in src/{names['import_name']}/__init__.py")
    print(f"  4. uv sync")
    
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
        help="Python version (default: 3.10)",
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
    init_model(args.name, args.python, args.torch, args.output, args.dry_run)


if __name__ == "__main__":
    main()
