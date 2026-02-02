"""GPT-SoVITS voice cloning TTS model.

GPT-SoVITS supports 4 major versions:
- v1: Original release (Jan 2024) - Chinese, Japanese, English
- v2: Extended release (Aug 2024) - +Korean, Cantonese, 5k hours training
- v3: DiT-based s2 (2024/2025) - 7k hours, improved zero-shot, 24k output
- v4: Custom vocoder (2025) - fixes v3 artifacts, 48k output
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

# Add vendored upstream code to sys.path
# Note: config.yaml specifies extra_paths: [GPT_SoVITS] which is also added
# to sys.path, enabling internal imports like `import utils` to work.
setup_vendor_path("ttsdb_gpt_sovits")

__all__ = ["GPTSoVITS"]

# Language mapping: ttsdb codes -> GPT-SoVITS codes
LANG_MAP = {
    "eng": "en",
    "en": "en",
    "zho": "zh",
    "zh": "zh",
    "jpn": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
    "yue": "yue",  # Cantonese
}

# Supported languages per version
VERSION_LANGUAGES = {
    "v1": {"en", "zh", "ja"},
    "v2": {"en", "zh", "ja", "ko", "yue"},
    "v3": {"en", "zh", "ja", "ko", "yue"},
    "v4": {"en", "zh", "ja", "ko", "yue"},
}


class DictToAttrRecursive(dict):
    """Recursively convert dict into an object with attribute-style access."""

    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"Attribute {item} not found") from e

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(f"Attribute {item} not found") from e


class GPTSoVITS(VoiceCloningTTSBase):
    """GPT-SoVITS voice cloning TTS model.

    A powerful few-shot voice conversion and text-to-speech system that achieves
    high-quality voice cloning with just 1 minute of training data.

    Supports multiple versions:
    - v1: Original release with Chinese, Japanese, English support
    - v2: Extended with Korean, Cantonese, and improved training
    - v3: DiT-based architecture with better zero-shot similarity
    - v4: Custom vocoder with 48k output

    Example:
        >>> model = GPTSoVITS(model_id="ttsds/gpt-sovits", variant="v2")
        >>> audio, sr = model.synthesize(
        ...     text="Hello world",
        ...     reference_audio="speaker.wav",
        ...     text_reference="This is reference text.",
        ...     language="en",
        ... )

    Args:
        model_path: Path to local model directory containing checkpoint files.
        model_id: HuggingFace model identifier.
        device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
        variant: Model variant to use ('v1', 'v2', 'v3', 'v4'). Default: 'v1'.
    """

    _package_name = "ttsdb_gpt_sovits"

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_id: str | None = None,
        device: str | torch.device | None = None,
        variant: str | None = None,
        **kwargs,
    ):
        self._version: str = "v1"
        self._is_half: bool = True
        self._weights_base: str | None = None
        self._sovits_config: Any = None
        self._max_sec: int = 54

        # Sub-models
        self.ssl_model = None  # cnhubert
        self.bert_model = None
        self.tokenizer = None
        self.vq_model = None  # SoVITS
        self.t2s_model = None  # GPT
        self.vocoder = None  # For v3/v4

        super().__init__(
            model_path=model_path,
            model_id=model_id,
            device=device,
            variant=variant,
            **kwargs,
        )

    def _iter_torch_modules(self) -> list[torch.nn.Module]:
        """Return all torch modules for .to()/.eval() handling."""
        modules = []
        for attr in ["ssl_model", "bert_model", "vq_model", "t2s_model", "vocoder"]:
            obj = getattr(self, attr, None)
            if isinstance(obj, torch.nn.Module):
                modules.append(obj)
        return modules

    def _load_model(self, load_path: str):
        """Load GPT-SoVITS models from the weights directory."""
        import librosa
        import soundfile as sf

        base = Path(load_path)
        if not base.exists():
            raise FileNotFoundError(
                f"Model path not found: {load_path}. "
                "Run `just weights gpt-sovits` to download weights first."
            )

        self._weights_base = str(base)

        # Determine version from variant or detect from files
        variant = self.model_config.variant if self.model_config else None
        self._version = str(self.init_kwargs.get("version") or variant or "v1")

        if self._version not in ("v1", "v2", "v3", "v4"):
            raise ValueError(f"Unknown version: {self._version}. Expected v1/v2/v3/v4")

        # Set precision
        self._is_half = self.init_kwargs.get("is_half", True) and self.device.type == "cuda"
        dtype = torch.float16 if self._is_half else torch.float32

        variant_dir = base / self._version
        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Variant directory not found: {variant_dir}. "
                f"Run `just weights gpt-sovits --variant {self._version}` to download."
            )

        shared_dir = base / "shared"

        # Set environment variable for patched code
        os.environ["TTSDB_WEIGHTS_DIR"] = str(base)

        # Some upstream utilities assume relative paths like
        # `GPT_SoVITS/text/G2PWModel` exist under the vendored repo root.
        # Create a link there pointing at our shared weights directory.
        try:
            from ttsdb_core import get_vendor_path

            vendor_root = get_vendor_path("ttsdb_gpt_sovits")
            g2pw_link = vendor_root / "GPT_SoVITS" / "text" / "G2PWModel"
            g2pw_target = shared_dir / "G2PWModel"
            if g2pw_target.exists() and not g2pw_link.exists():
                g2pw_link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    g2pw_link.symlink_to(g2pw_target, target_is_directory=True)
                except OSError:
                    import shutil

                    shutil.copytree(g2pw_target, g2pw_link)
        except Exception:
            # If this fails, upstream may attempt to download G2PW at runtime.
            pass

        # 1) Load BERT for text features
        self._load_bert(shared_dir, dtype)

        # 2) Load cnhubert for audio features
        self._load_cnhubert(shared_dir, dtype)

        # 3) Load SoVITS and GPT models
        if self._version in ("v1", "v2"):
            self._load_sovits_v1v2(variant_dir, dtype)
        else:
            self._load_sovits_v3v4(variant_dir, dtype)

        print(f"Loaded GPT-SoVITS {self._version}")
        return self.vq_model

    def _load_bert(self, shared_dir: Path, dtype: torch.dtype) -> None:
        """Load Chinese BERT model for text features."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        bert_path = shared_dir / "chinese-roberta-wwm-ext-large"
        if not bert_path.exists():
            raise FileNotFoundError(f"BERT model not found at {bert_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
        self.bert_model = AutoModelForMaskedLM.from_pretrained(str(bert_path))

        if self._is_half:
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()

    def _load_cnhubert(self, shared_dir: Path, dtype: torch.dtype) -> None:
        """Load Chinese HuBERT model for audio feature extraction."""
        from GPT_SoVITS.feature_extractor import cnhubert

        hubert_path = shared_dir / "chinese-hubert-base"
        if not hubert_path.exists():
            raise FileNotFoundError(f"cnhubert not found at {hubert_path}")

        cnhubert.cnhubert_base_path = str(hubert_path)
        self.ssl_model = cnhubert.get_model()

        if self._is_half:
            self.ssl_model = self.ssl_model.half()
        self.ssl_model = self.ssl_model.to(self.device)
        self.ssl_model.eval()

    def _load_sovits_v1v2(self, variant_dir: Path, dtype: torch.dtype) -> None:
        """Load SoVITS v1/v2 models (VITS-based)."""
        from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
        from GPT_SoVITS.module.models import SynthesizerTrn

        # Find checkpoint files
        sovits_path = variant_dir / "sovits.pth"
        gpt_path = variant_dir / "gpt.ckpt"

        if not sovits_path.exists():
            raise FileNotFoundError(f"SoVITS checkpoint not found: {sovits_path}")
        if not gpt_path.exists():
            raise FileNotFoundError(f"GPT checkpoint not found: {gpt_path}")

        # Load SoVITS
        checkpoint = torch.load(str(sovits_path), map_location="cpu", weights_only=False)
        self._sovits_config = DictToAttrRecursive(checkpoint["config"])

        # Detect version from embedding shape
        emb = checkpoint["weight"]["enc_p.text_embedding.weight"]
        if emb.shape[0] == 322:
            self._sovits_config.model.version = "v1"
        else:
            self._sovits_config.model.version = "v2"

        self._sovits_config.model.semantic_frame_rate = "25hz"

        self.vq_model = SynthesizerTrn(
            self._sovits_config.data.filter_length // 2 + 1,
            self._sovits_config.train.segment_size // self._sovits_config.data.hop_length,
            n_speakers=self._sovits_config.data.n_speakers,
            **self._sovits_config.model,
        )

        # Remove enc_q if not pretrained
        if hasattr(self.vq_model, "enc_q"):
            del self.vq_model.enc_q

        self.vq_model.load_state_dict(checkpoint["weight"], strict=False)
        if self._is_half:
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()

        # Load GPT
        gpt_ckpt = torch.load(str(gpt_path), map_location="cpu", weights_only=False)
        self._gpt_config = gpt_ckpt["config"]
        self._max_sec = self._gpt_config["data"]["max_sec"]

        self.t2s_model = Text2SemanticLightningModule(self._gpt_config, "dummy", is_train=False)
        self.t2s_model.load_state_dict(gpt_ckpt["weight"])
        if self._is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        total_params = sum(p.numel() for p in self.t2s_model.parameters())
        print(f"  GPT params: {total_params / 1e6:.1f}M")

    def _load_sovits_v3v4(self, variant_dir: Path, dtype: torch.dtype) -> None:
        """Load SoVITS v3/v4 models (DiT-based)."""
        # v3/v4 use different architecture - import the appropriate modules
        # This is a placeholder - actual implementation depends on upstream code structure
        from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

        sovits_path = variant_dir / "sovits.pth"
        gpt_path = variant_dir / "gpt.ckpt"

        if not sovits_path.exists():
            raise FileNotFoundError(f"SoVITS checkpoint not found: {sovits_path}")
        if not gpt_path.exists():
            raise FileNotFoundError(f"GPT checkpoint not found: {gpt_path}")

        # Load GPT (same structure as v1/v2)
        gpt_ckpt = torch.load(str(gpt_path), map_location="cpu", weights_only=False)
        self._gpt_config = gpt_ckpt["config"]
        self._max_sec = self._gpt_config["data"]["max_sec"]

        self.t2s_model = Text2SemanticLightningModule(self._gpt_config, "dummy", is_train=False)
        self.t2s_model.load_state_dict(gpt_ckpt["weight"])
        if self._is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        # Load SoVITS v3/v4 (DiT-based)
        # SynthesizerTrnV3 is in module.models, not module.models_s2
        from GPT_SoVITS.module.models import SynthesizerTrnV3

        checkpoint = torch.load(str(sovits_path), map_location="cpu", weights_only=False)
        self._sovits_config = DictToAttrRecursive(checkpoint["config"])
        self._sovits_config.model.version = self._version

        self.vq_model = SynthesizerTrnV3(
            self._sovits_config.data.filter_length // 2 + 1,
            self._sovits_config.train.segment_size // self._sovits_config.data.hop_length,
            n_speakers=self._sovits_config.data.n_speakers,
            **self._sovits_config.model,
        )
        # Load weights, removing enc_q if present (not needed for inference)
        if hasattr(self.vq_model, "enc_q"):
            del self.vq_model.enc_q
        self.vq_model.load_state_dict(checkpoint["weight"], strict=False)

        if self._is_half:
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()

        # Load vocoder for v3/v4
        if self._version == "v3":
            bigvgan_dir = variant_dir / "bigvgan"
            if bigvgan_dir.exists():
                self._load_bigvgan(bigvgan_dir)
        elif self._version == "v4":
            vocoder_path = variant_dir / "vocoder.pth"
            if vocoder_path.exists():
                self._load_vocoder_v4(vocoder_path)

        total_params = sum(p.numel() for p in self.t2s_model.parameters())
        print(f"  GPT params: {total_params / 1e6:.1f}M")

    def _load_bigvgan(self, bigvgan_dir: Path) -> None:
        """Load BigVGAN vocoder for v3."""
        try:
            from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN

            self.vocoder = BigVGAN.from_pretrained(
                str(bigvgan_dir),
                use_cuda_kernel=False,
            )
            self.vocoder.remove_weight_norm()
            if self._is_half:
                self.vocoder = self.vocoder.half()
            self.vocoder = self.vocoder.to(self.device)
            self.vocoder.eval()
            print(f"  ✓ BigVGAN vocoder loaded from {bigvgan_dir}")
        except Exception as e:
            print(f"  ✗ ERROR: Could not load BigVGAN vocoder: {e}")
            import traceback

            traceback.print_exc()
            self.vocoder = None

    def _load_vocoder_v4(self, vocoder_path: Path) -> None:
        """Load custom vocoder for v4."""
        try:
            from GPT_SoVITS.module.models import Generator

            self.vocoder = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],  # 10*6*2*2*2 = 480
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            # IMPORTANT: Remove weight norm BEFORE loading state_dict.
            # The saved weights are from a model without weight norm applied,
            # so we need to convert our initialized model to match that structure.
            self.vocoder.remove_weight_norm()
            state_dict = torch.load(str(vocoder_path), map_location="cpu", weights_only=False)
            self.vocoder.load_state_dict(state_dict)
            if self._is_half:
                self.vocoder = self.vocoder.half()
            self.vocoder = self.vocoder.to(self.device)
            self.vocoder.eval()
            print(f"  ✓ V4 vocoder loaded from {vocoder_path}")
        except Exception as e:
            print(f"  ✗ ERROR: Could not load v4 vocoder: {e}")
            import traceback

            traceback.print_exc()
            self.vocoder = None

    def _get_phones_and_bert(
        self,
        text: str,
        language: str,
    ) -> tuple[list[int], torch.Tensor, str]:
        """Convert text to phone IDs and BERT features."""
        import LangSegment
        from GPT_SoVITS.text import chinese, cleaned_text_to_sequence
        from GPT_SoVITS.text.cleaner import clean_text

        dtype = torch.float16 if self._is_half else torch.float32
        version = self._sovits_config.model.version

        lang = language
        if lang == "en":
            # g2p_en uses NLTK taggers; ensure required resources exist.
            try:
                import nltk

                try:
                    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
                except LookupError:
                    # Newer NLTK tagger name
                    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

                # Some environments still expect the legacy resource name.
                try:
                    nltk.data.find("taggers/averaged_perceptron_tagger")
                except LookupError:
                    nltk.download("averaged_perceptron_tagger", quiet=True)
            except Exception:
                # If downloads are unavailable, the downstream call may raise a
                # clearer error; don't hide it here.
                pass

            # LangSegment API differs between upstream and the PyPI
            # `langsegment-backup` package. Support both.
            if hasattr(LangSegment, "setLangfilters"):
                LangSegment.setLangfilters(["en"])
            elif hasattr(LangSegment, "setfilters"):
                LangSegment.setfilters(["en"])
            formattext = " ".join(seg["text"] for seg in LangSegment.getTexts(text))
        else:
            formattext = text

        # Clean whitespace
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")

        if lang == "zh":
            if re.search(r"[a-zA-Z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                normalize = getattr(chinese, "mix_text_normalize", None) or chinese.text_normalize
                formattext = normalize(formattext)
                return self._get_phones_and_bert(formattext, "zh")

            phones, word2ph, norm_text = self._clean_text_inf(formattext, lang, version)
            bert = self._get_bert_feature(norm_text, word2ph)
        elif lang == "yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            normalize = getattr(chinese, "mix_text_normalize", None) or chinese.text_normalize
            formattext = normalize(formattext)
            return self._get_phones_and_bert(formattext, "yue")
        else:
            phones, word2ph, norm_text = self._clean_text_inf(formattext, lang, version)
            # Non-Chinese: zero BERT
            bert = torch.zeros((1024, len(phones)), dtype=dtype).to(self.device)

        return phones, bert, norm_text

    def _clean_text_inf(
        self, text: str, language: str, version: str
    ) -> tuple[list[int], list[int], str]:
        """Clean text and convert to phone IDs."""
        from GPT_SoVITS.text import cleaned_text_to_sequence
        from GPT_SoVITS.text.cleaner import clean_text

        # The upstream text pipeline (notably G2PW) uses relative paths and may
        # try to write under `GPT_SoVITS/text/...`. Run it from the vendored repo
        # root so those paths resolve correctly.
        from ttsdb_core import vendor_context

        with vendor_context("ttsdb_gpt_sovits", cwd=True):
            phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def _get_bert_feature(self, text: str, word2ph: list[int]) -> torch.Tensor:
        """Extract BERT features at phone level."""
        # Some lightweight installs may omit the Chinese BERT dependency/weights.
        # In that case, fall back to zero conditioning (works for non-Chinese
        # paths; Chinese quality will be degraded but synthesis can proceed).
        if self.tokenizer is None or self.bert_model is None:
            dtype = torch.float16 if self._is_half else torch.float32
            return torch.zeros((1024, sum(word2ph)), dtype=dtype).to(self.device)

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(
            text
        ), f"word2ph length {len(word2ph)} != text length {len(text)}"
        phone_level_features = []
        for i in range(len(word2ph)):
            repeated_feature = res[i].repeat(word2ph[i], 1)
            phone_level_features.append(repeated_feature)
        phone_level_features = torch.cat(phone_level_features, dim=0)
        return phone_level_features.T

    def _get_spepc(self, audio_path: str) -> torch.Tensor:
        """Compute spectrogram for reference audio."""
        from GPT_SoVITS.module.mel_processing import spectrogram_torch

        hps = self._sovits_config
        # Avoid importing upstream `tools.my_utils` (it pulls in gradio/UI deps).
        # We only need a minimal audio loader + resampler here.
        import librosa
        import soundfile as sf

        target_sr = int(hps.data.sampling_rate)
        audio, sr = sf.read(audio_path, always_2d=False)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            # Convert to mono
            audio = audio.mean(axis=1)
        if int(sr) != target_sr:
            audio = librosa.resample(audio.astype("float32"), orig_sr=int(sr), target_sr=target_sr)
        audio = torch.FloatTensor(audio)

        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)

        audio = audio.unsqueeze(0)
        spec = spectrogram_torch(
            audio,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )

        if self._is_half:
            return spec.half().to(self.device)
        return spec.to(self.device)

    def _synthesize(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        text_reference: str = "",
        language: str = "en",
        **kwargs,
    ) -> AudioOutput:
        """Synthesize speech from text using GPT-SoVITS.

        Args:
            text: Input text to synthesize.
            reference_audio: Reference audio as numpy array.
            reference_sample_rate: Sample rate of reference audio.
            text_reference: Transcript of the reference audio.
            language: Language code ('en', 'zh', 'ja', 'ko', 'yue').
            **kwargs: Additional parameters:
                - top_k: GPT sampling top-k (default: 15)
                - top_p: GPT sampling top-p (default: 1.0)
                - temperature: GPT sampling temperature (default: 1.0)
                - speed: Speech speed multiplier (default: 1.0)
                - ref_free: If True, ignore reference text (default: False)

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        import librosa
        import soundfile as sf

        # Map language code
        lang = LANG_MAP.get(language, language)
        supported = VERSION_LANGUAGES.get(self._version, {"en", "zh", "ja"})
        if lang not in supported:
            raise ValueError(
                f"Language '{lang}' not supported in {self._version}. " f"Supported: {supported}"
            )

        # Sampling parameters
        top_k = int(kwargs.get("top_k", 15))
        top_p = float(kwargs.get("top_p", 1.0))
        temperature = float(kwargs.get("temperature", 1.0))
        speed = float(kwargs.get("speed", 1.0))
        ref_free = bool(kwargs.get("ref_free", False))

        dtype = torch.float16 if self._is_half else torch.float32
        # Output sample rate: v1/v2 from config (32k), v3=24k, v4=48k
        # Note: v4 mel is computed at 32kHz with hop=320 (100 frames/sec),
        # vocoder upsamples by 480x, so output is 100*480=48kHz
        if self._version == "v3":
            sr_out = 24000
        elif self._version == "v4":
            sr_out = 48000
        else:
            sr_out = int(self._sovits_config.data.sampling_rate)

        # Save reference audio to temp file for loading
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, reference_audio, reference_sample_rate)

        try:
            # 1) Extract semantic codes from reference audio
            wav16k, _ = librosa.load(tmp_path, sr=16000)
            wav16k_torch = torch.from_numpy(wav16k).to(self.device, dtype=dtype)

            # Add padding
            pad = torch.zeros(int(0.3 * 16000), dtype=dtype, device=self.device)
            wav16k_torch = torch.cat([wav16k_torch, pad], dim=0)

            with torch.no_grad():
                ssl_content = self.ssl_model.model(wav16k_torch.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0].unsqueeze(0)

            # 2) Process prompt text
            if not ref_free and text_reference:
                phones_prompt, bert_prompt, _ = self._get_phones_and_bert(text_reference, lang)
            else:
                phones_prompt = []
                bert_prompt = torch.zeros((1024, 0), dtype=dtype, device=self.device)

            # 3) Process target text (split by newlines)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if not lines:
                lines = [text]

            # Merge short lines
            lines = self._merge_short_texts(lines, threshold=5)

            # Output audio segments
            zero_pause = np.zeros(int(sr_out * 0.3), dtype=np.float32)
            output_audio = []

            # Get reference spectrogram
            ref_spec = self._get_spepc(tmp_path)

            for line in lines:
                phones_line, bert_line, norm_text = self._get_phones_and_bert(line, lang)

                # Merge with prompt if not ref_free
                if not ref_free:
                    all_phones = phones_prompt + phones_line
                    all_bert = torch.cat([bert_prompt, bert_line], dim=1)
                else:
                    all_phones = phones_line
                    all_bert = bert_line

                phone_ids = torch.LongTensor(all_phones).unsqueeze(0).to(self.device)
                bert_input = all_bert.unsqueeze(0).to(self.device)
                phone_len = torch.tensor([phone_ids.shape[-1]]).to(self.device)

                # GPT inference
                with torch.no_grad():
                    pred_semantic, used_length = self.t2s_model.model.infer_panel(
                        phone_ids,
                        phone_len,
                        prompt_semantic if not ref_free else None,
                        bert_input,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=50 * self._max_sec,
                    )
                    if not ref_free:
                        pred_semantic = pred_semantic[:, -used_length:].unsqueeze(0)
                    else:
                        pred_semantic = pred_semantic.unsqueeze(0)

                # SoVITS decode - different path for v3/v4 vs v1/v2
                if self._version in ("v1", "v2"):
                    # Direct waveform decode for v1/v2
                    audio_out = self.vq_model.decode(
                        pred_semantic,
                        torch.LongTensor(phones_line).unsqueeze(0).to(self.device),
                        [ref_spec],
                        speed=speed,
                    )
                    audio_np = audio_out.detach().cpu().numpy()[0, 0]
                else:
                    # v3/v4: Use CFM + vocoder pipeline
                    audio_np = self._decode_v3v4(
                        pred_semantic,
                        phones_prompt,
                        phones_line,
                        prompt_semantic,
                        ref_spec,
                        tmp_path,
                        speed,
                    )

                # Normalize
                max_amp = np.abs(audio_np).max()
                if max_amp > 1.0:
                    audio_np /= max_amp

                output_audio.append(audio_np.astype(np.float32))
                output_audio.append(zero_pause)

            # Concatenate all segments
            final_wav = np.concatenate(output_audio, axis=0)
            return final_wav, sr_out

        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _decode_v3v4(
        self,
        pred_semantic: torch.Tensor,
        phones_prompt: list[int],
        phones_line: list[int],
        prompt_semantic: torch.Tensor,
        ref_spec: torch.Tensor,
        ref_audio_path: str,
        speed: float,
        sample_steps: int = 32,
    ) -> np.ndarray:
        """Decode using v3/v4 CFM + vocoder pipeline.

        v3/v4 models use a different architecture:
        1. decode_encp extracts features
        2. CFM model converts to mel spectrogram
        3. Vocoder (BigVGAN/Generator) converts mel to waveform
        """
        import librosa
        import soundfile as sf
        from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch

        dtype = torch.float16 if self._is_half else torch.float32

        # Load reference audio using soundfile/librosa (avoids torchcodec dependency)
        ref_audio_np, ref_sr = sf.read(ref_audio_path)
        if ref_audio_np.ndim > 1:
            ref_audio_np = ref_audio_np.mean(axis=1)  # Convert to mono
        ref_audio = torch.from_numpy(ref_audio_np).unsqueeze(0).to(self.device).to(dtype)

        # Target sample rate for mel computation: 24k for v3, 32k for v4
        # (Note: v4 vocoder output is 32kHz, not 48kHz as originally thought)
        tgt_sr = 24000 if self._version == "v3" else 32000
        if ref_sr != tgt_sr:
            # Resample using librosa
            ref_audio_np = ref_audio[0].cpu().numpy()
            ref_audio_np = librosa.resample(ref_audio_np, orig_sr=ref_sr, target_sr=tgt_sr)
            ref_audio = torch.from_numpy(ref_audio_np).unsqueeze(0).to(self.device).to(dtype)

        # Compute mel spectrogram of reference audio
        if self._version == "v3":
            mel2 = mel_spectrogram_torch(
                ref_audio,
                n_fft=1024,
                win_size=1024,
                hop_size=256,
                num_mels=100,
                sampling_rate=24000,
                fmin=0,
                fmax=None,
                center=False,
            )
        else:  # v4
            mel2 = mel_spectrogram_torch(
                ref_audio,
                n_fft=1280,
                win_size=1280,
                hop_size=320,
                num_mels=100,
                sampling_rate=32000,
                fmin=0,
                fmax=None,
                center=False,
            )

        # Normalize mel
        spec_min, spec_max = -12, 2
        mel2 = (mel2 - spec_min) / (spec_max - spec_min) * 2 - 1

        # Prepare inputs
        prompt_phones_t = torch.LongTensor(phones_prompt).unsqueeze(0).to(self.device)
        target_phones_t = torch.LongTensor(phones_line).unsqueeze(0).to(self.device)
        # semantic codes need to be 3D: [num_quantizers, batch, seq_len]
        # prompt_semantic from _synthesize is [1, seq_len] (2D), needs unsqueeze to get [1, 1, seq_len]
        # pred_semantic from _synthesize is already [1, 1, seq_len] (3D) after unsqueeze(0) in _synthesize
        prompt_semantic_t = prompt_semantic.unsqueeze(0).to(self.device)  # [1, 1, seq_len]
        pred_semantic_t = pred_semantic.to(self.device)  # Already [1, 1, seq_len]

        with torch.no_grad():
            # Get reference features using prompt semantic codes
            fea_ref, ge = self.vq_model.decode_encp(prompt_semantic_t, prompt_phones_t, ref_spec)

            # Get target features using predicted semantic codes
            fea_todo, ge = self.vq_model.decode_encp(
                pred_semantic_t, target_phones_t, ref_spec, ge, speed
            )

            # Align lengths
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min].to(dtype)
            fea_ref = fea_ref[:, :, :T_min]

            # CFM parameters
            T_ref = 468 if self._version == "v3" else 500
            T_chunk = 934 if self._version == "v3" else 1000

            if T_min > T_ref:
                mel2 = mel2[:, :, -T_ref:]
                fea_ref = fea_ref[:, :, -T_ref:]
                T_min = T_ref

            chunk_len = T_chunk - T_min

            # Process in chunks through CFM
            cfm_results = []
            idx = 0
            while True:
                fea_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_chunk.shape[-1] == 0:
                    break
                idx += chunk_len

                fea = torch.cat([fea_ref, fea_chunk], 2).transpose(2, 1)
                cfm_out = self.vq_model.cfm.inference(
                    fea,
                    torch.LongTensor([fea.size(1)]).to(self.device),
                    mel2,
                    sample_steps,
                    inference_cfg_rate=0,
                )
                cfm_out = cfm_out[:, :, mel2.shape[2] :]

                # Update reference for next chunk
                mel2 = cfm_out[:, :, -T_min:]
                fea_ref = fea_chunk[:, :, -T_min:]

                cfm_results.append(cfm_out)

            # Concatenate all CFM outputs
            cfm_res = torch.cat(cfm_results, 2)

            # Denormalize mel
            cfm_res = (cfm_res + 1) / 2 * (spec_max - spec_min) + spec_min

            # Generate waveform with vocoder
            if self.vocoder is not None:
                wav_gen = self.vocoder(cfm_res)
                audio_np = wav_gen[0, 0].detach().cpu().numpy()
            else:
                # Fallback: return silence if vocoder not loaded
                raise RuntimeError(
                    f"Vocoder not loaded for {self._version}! Cannot generate audio. "
                    "Check vocoder loading errors above."
                )
                audio_np = np.zeros(int(cfm_res.shape[2] * 256), dtype=np.float32)

        return audio_np

    def _merge_short_texts(self, texts: list[str], threshold: int = 5) -> list[str]:
        """Merge short text segments."""
        if len(texts) < 2:
            return texts
        result = []
        tmp = ""
        for txt in texts:
            tmp += txt
            if len(tmp) >= threshold:
                result.append(tmp)
                tmp = ""
        if tmp:
            if not result:
                result = [tmp]
            else:
                result[-1] += tmp
        return result
