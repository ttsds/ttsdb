#!/usr/bin/env python3
"""Patch vendored Fish Speech code to keep runtime deps minimal.

Called automatically by `python builder/vendor.py models/fish-speech` if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PATCH_BLOCK = """
try:
    from audiotools import AudioSignal
    from audiotools.ml import BaseModel
except Exception:  # pragma: no cover - optional dependency for training
    AudioSignal = None

    class BaseModel(nn.Module):
        pass
""".lstrip()

FIREFLY_GAN_VQ_YAML = """
_target_: fish_speech.models.vqgan.modules.firefly.FireflyArchitecture
spec_transform:
    _target_: fish_speech.utils.spectrogram.LogMelSpectrogram
    sample_rate: 44100
    n_mels: 160
    n_fft: 2048
    hop_length: 512
    win_length: 2048
backbone:
    _target_: fish_speech.models.vqgan.modules.firefly.ConvNeXtEncoder
    input_channels: 160
    depths: [3, 3, 9, 3]
    dims: [128, 256, 384, 512]
    drop_path_rate: 0.2
    kernel_size: 7
head:
    _target_: fish_speech.models.vqgan.modules.firefly.HiFiGANGenerator
    hop_length: 512
    upsample_rates: [8, 8, 2, 2, 2]  # aka. strides
    upsample_kernel_sizes: [16, 16, 4, 4, 4]
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    num_mels: 512
    upsample_initial_channel: 512
    pre_conv_kernel_size: 13
    post_conv_kernel_size: 13
quantizer:
    _target_: fish_speech.models.vqgan.modules.fsq.DownsampleFiniteScalarQuantize
    input_dim: 512
    n_groups: 8
    n_codebooks: 1
    levels: [8, 5, 5, 5]
    downsample_factor: [2, 2]
""".lstrip()


def _patch_file(path: Path) -> None:
    s = path.read_text()

    if "from audiotools import AudioSignal" in s and "try:\n    from audiotools" in s:
        return

    s = s.replace("import librosa\n", "")
    s = s.replace("from audiotools import AudioSignal\n", "")
    s = s.replace("from audiotools.ml import BaseModel\n", "")

    marker = "from torch.nn.utils.parametrize import remove_parametrizations\n\n\n@dataclass"
    if marker in s:
        s = s.replace(
            marker,
            f"from torch.nn.utils.parametrize import remove_parametrizations\n\n{PATCH_BLOCK}\n\n@dataclass",
        )
    else:
        # Fallback: insert before first dataclass if marker changed upstream.
        s = s.replace("\n\n@dataclass", f"\n\n{PATCH_BLOCK}\n\n@dataclass", 1)

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-root", required=True, type=Path)
    ap.add_argument("--config", required=False, type=Path)
    args = ap.parse_args()

    vendor_root: Path = args.vendor_root

    target = vendor_root / "fish_speech" / "models" / "dac" / "modded_dac.py"
    if not target.exists():
        matches = list(vendor_root.rglob("modded_dac.py"))
        if matches:
            target = matches[0]
        else:
            raise FileNotFoundError("Could not find modded_dac.py in vendored fish-speech repo")

    _patch_file(target)
    print(f"Patched {target}")

    configs_dir = vendor_root / "fish_speech" / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    firefly_config_path = configs_dir / "firefly_gan_vq.yaml"
    current = firefly_config_path.read_text() if firefly_config_path.exists() else None
    if current != FIREFLY_GAN_VQ_YAML:
        firefly_config_path.write_text(FIREFLY_GAN_VQ_YAML)
        print(f"Wrote {firefly_config_path}")


if __name__ == "__main__":
    main()
