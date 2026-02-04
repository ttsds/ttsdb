#!/usr/bin/env python3
"""Patch vendored VoiceCraft source with extra dependencies.

Currently adds the audiocraft dependency as a local vendored copy so that
AudioTokenizer can import it without pip-installed extras.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

AUDIOCRAFT_REPO = "https://github.com/facebookresearch/audiocraft.git"
AUDIOCRAFT_COMMIT = "f83babff6b5e97f75562127c4cc8122229c8f099"


def _clone_repo(repo_url: str, commit: str, target_dir: Path) -> None:
    if target_dir.exists():
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", commit],
        cwd=target_dir,
        check=True,
    )
    subprocess.run(["git", "checkout", commit], cwd=target_dir, check=True)

    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True, help="Path to vendored source root")
    parser.add_argument("--config", required=False, help="Path to config.yaml (unused)")
    args = parser.parse_args()

    vendor_root = Path(args.vendor_root).resolve()
    audiocraft_dir = vendor_root / "audiocraft"
    _clone_repo(AUDIOCRAFT_REPO, AUDIOCRAFT_COMMIT, audiocraft_dir)

    tokenizer_path = vendor_root / "data" / "tokenizer.py"
    if tokenizer_path.exists():
        text = tokenizer_path.read_text()

        import_block = "import numpy as np\nimport torch\nimport torchaudio\n"
        extra_imports = "from pathlib import Path\nimport sys\nimport importlib.util\n"
        if extra_imports not in text:
            text = text.replace(import_block, import_block + extra_imports)

        if "def _ensure_audiocraft_vendor" not in text:
            matcher = re.search(r"class AudioTokenizer:\n(\s+\"\"\"EnCodec audio\.\"\"\")", text)
            if matcher:
                insert_at = matcher.end(0)
                helper = (
                    "\n\n    def _ensure_audiocraft_vendor(self) -> None:\n"
                    "        vendor_root = Path(__file__).resolve().parents[2]\n"
                    '        audiocraft_repo = vendor_root / "audiocraft"\n'
                    '        audiocraft_pkg = audiocraft_repo / "audiocraft"\n'
                    "        if not audiocraft_pkg.exists():\n"
                    "            return\n\n"
                    "        repo_str = str(audiocraft_repo)\n"
                    "        vendor_str = str(vendor_root)\n"
                    "        if repo_str not in sys.path:\n"
                    "            sys.path.insert(0, repo_str)\n"
                    "        if vendor_str not in sys.path:\n"
                    "            sys.path.insert(0, vendor_str)\n\n"
                    '        existing = sys.modules.get("audiocraft")\n'
                    "        if existing is not None:\n"
                    '            module_path = getattr(existing, "__file__", "") or ""\n'
                    "            if repo_str not in module_path:\n"
                    "                for key in list(sys.modules.keys()):\n"
                    '                    if key == "audiocraft" or key.startswith("audiocraft."):\n'
                    "                        del sys.modules[key]\n\n"
                    '        if "audiocraft" not in sys.modules:\n'
                    "            spec = importlib.util.spec_from_file_location(\n"
                    '                "audiocraft",\n'
                    '                audiocraft_pkg / "__init__.py",\n'
                    "                submodule_search_locations=[str(audiocraft_pkg)],\n"
                    "            )\n"
                    "            if spec and spec.loader:\n"
                    "                module = importlib.util.module_from_spec(spec)\n"
                    '                sys.modules["audiocraft"] = module\n'
                    "                spec.loader.exec_module(module)\n\n"
                    '        solvers_pkg = audiocraft_pkg / "solvers"\n'
                    "        if solvers_pkg.exists():\n"
                    "            solvers_spec = importlib.util.spec_from_file_location(\n"
                    '                "audiocraft.solvers",\n'
                    '                solvers_pkg / "__init__.py",\n'
                    "                submodule_search_locations=[str(solvers_pkg)],\n"
                    "            )\n"
                    "            if solvers_spec and solvers_spec.loader:\n"
                    "                solvers_module = importlib.util.module_from_spec(solvers_spec)\n"
                    '                sys.modules["audiocraft.solvers"] = solvers_module\n'
                    "                solvers_spec.loader.exec_module(solvers_module)\n\n"
                    '            compression_path = solvers_pkg / "compression.py"\n'
                    "            if compression_path.exists():\n"
                    "                compression_spec = importlib.util.spec_from_file_location(\n"
                    '                    "audiocraft.solvers.compression",\n'
                    "                    compression_path,\n"
                    "                )\n"
                    "                if compression_spec and compression_spec.loader:\n"
                    "                    compression_module = importlib.util.module_from_spec(compression_spec)\n"
                    '                    sys.modules["audiocraft.solvers.compression"] = compression_module\n'
                    "                    compression_spec.loader.exec_module(compression_module)\n"
                )
                text = text[:insert_at] + helper + text[insert_at:]

        if "def __init__(" in text and "ac_checkpoint.resolve_checkpoint_path" not in text:
            text = text.replace(
                "def __init__(\n        self,\n        device: Any = None,\n        signature = None\n    ) -> None:\n",
                "def __init__(\n        self,\n        device: Any = None,\n        signature = None\n    ) -> None:\n        self._ensure_audiocraft_vendor()\n",
            )
            text = text.replace(
                "from audiocraft.solvers import CompressionSolver\n        model = CompressionSolver.model_from_checkpoint(signature)",
                "from audiocraft.solvers import CompressionSolver\n        from audiocraft.utils import checkpoint as ac_checkpoint\n\n        def _resolve_checkpoint_path(sig_or_path, name=None, use_fsdp: bool = False):\n            path = Path(str(sig_or_path))\n            if path.is_dir():\n                path = path / ac_checkpoint.checkpoint_name(name, use_fsdp=use_fsdp)\n            return path if path.exists() else None\n\n        ac_checkpoint.resolve_checkpoint_path = _resolve_checkpoint_path\n        model = CompressionSolver.model_from_checkpoint(signature)",
            )

        tokenizer_path.write_text(text)


if __name__ == "__main__":
    main()
