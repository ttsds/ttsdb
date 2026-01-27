"""ISO 639-3 language codes for TTS models.

This module provides language code validation and name lookup based on ISO 639-3.
Reference: https://iso639-3.sil.org/

We maintain a curated subset of languages commonly used in TTS systems.
The full standard contains 8000+ language codes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Language:
    """A language with ISO 639-3 code and metadata."""
    code: str  # ISO 639-3 three-letter code
    name: str  # English name
    native_name: str | None = None  # Native name
    iso639_1: str | None = None  # ISO 639-1 two-letter code (if exists)


# Curated list of languages commonly used in TTS
# ISO 639-3 codes from https://iso639-3.sil.org/
LANGUAGES: dict[str, Language] = {
    # Major world languages
    "eng": Language("eng", "English", "English", "en"),
    "zho": Language("zho", "Chinese", "中文", "zh"),
    "cmn": Language("cmn", "Mandarin Chinese", "普通话"),  # More specific
    "yue": Language("yue", "Cantonese", "粵語"),
    "jpn": Language("jpn", "Japanese", "日本語", "ja"),
    "kor": Language("kor", "Korean", "한국어", "ko"),
    "fra": Language("fra", "French", "Français", "fr"),
    "deu": Language("deu", "German", "Deutsch", "de"),
    "spa": Language("spa", "Spanish", "Español", "es"),
    "ita": Language("ita", "Italian", "Italiano", "it"),
    "por": Language("por", "Portuguese", "Português", "pt"),
    "rus": Language("rus", "Russian", "Русский", "ru"),
    "ara": Language("ara", "Arabic", "العربية", "ar"),
    "hin": Language("hin", "Hindi", "हिन्दी", "hi"),
    "ben": Language("ben", "Bengali", "বাংলা", "bn"),
    "pan": Language("pan", "Punjabi", "ਪੰਜਾਬੀ", "pa"),
    "tam": Language("tam", "Tamil", "தமிழ்", "ta"),
    "tel": Language("tel", "Telugu", "తెలుగు", "te"),
    "mar": Language("mar", "Marathi", "मराठी", "mr"),
    "guj": Language("guj", "Gujarati", "ગુજરાતી", "gu"),
    "kan": Language("kan", "Kannada", "ಕನ್ನಡ", "kn"),
    "mal": Language("mal", "Malayalam", "മലയാളം", "ml"),
    "urd": Language("urd", "Urdu", "اردو", "ur"),
    
    # European languages
    "nld": Language("nld", "Dutch", "Nederlands", "nl"),
    "pol": Language("pol", "Polish", "Polski", "pl"),
    "tur": Language("tur", "Turkish", "Türkçe", "tr"),
    "vie": Language("vie", "Vietnamese", "Tiếng Việt", "vi"),
    "tha": Language("tha", "Thai", "ไทย", "th"),
    "ind": Language("ind", "Indonesian", "Bahasa Indonesia", "id"),
    "msa": Language("msa", "Malay", "Bahasa Melayu", "ms"),
    "swe": Language("swe", "Swedish", "Svenska", "sv"),
    "dan": Language("dan", "Danish", "Dansk", "da"),
    "nor": Language("nor", "Norwegian", "Norsk", "no"),
    "fin": Language("fin", "Finnish", "Suomi", "fi"),
    "ces": Language("ces", "Czech", "Čeština", "cs"),
    "ell": Language("ell", "Greek", "Ελληνικά", "el"),
    "heb": Language("heb", "Hebrew", "עברית", "he"),
    "ukr": Language("ukr", "Ukrainian", "Українська", "uk"),
    "ron": Language("ron", "Romanian", "Română", "ro"),
    "hun": Language("hun", "Hungarian", "Magyar", "hu"),
    "bul": Language("bul", "Bulgarian", "Български", "bg"),
    "hrv": Language("hrv", "Croatian", "Hrvatski", "hr"),
    "slk": Language("slk", "Slovak", "Slovenčina", "sk"),
    "slv": Language("slv", "Slovenian", "Slovenščina", "sl"),
    "est": Language("est", "Estonian", "Eesti", "et"),
    "lav": Language("lav", "Latvian", "Latviešu", "lv"),
    "lit": Language("lit", "Lithuanian", "Lietuvių", "lt"),
    "fas": Language("fas", "Persian", "فارسی", "fa"),
    "cat": Language("cat", "Catalan", "Català", "ca"),
}

# Build reverse lookup from ISO 639-1 to ISO 639-3
_ISO639_1_TO_3: dict[str, str] = {
    lang.iso639_1: code 
    for code, lang in LANGUAGES.items() 
    if lang.iso639_1
}


def get_language(code: str) -> Language | None:
    """Get a language by ISO 639-3 or ISO 639-1 code.
    
    Args:
        code: Language code (either ISO 639-1 or ISO 639-3).
        
    Returns:
        Language object if found, None otherwise.
    """
    # Try ISO 639-3 first
    if code in LANGUAGES:
        return LANGUAGES[code]
    
    # Try ISO 639-1 to ISO 639-3 conversion
    if code in _ISO639_1_TO_3:
        return LANGUAGES[_ISO639_1_TO_3[code]]
    
    return None


def get_language_name(code: str) -> str:
    """Get the English name for a language code.
    
    Args:
        code: Language code (ISO 639-1 or ISO 639-3).
        
    Returns:
        Language name, or the original code if not found.
    """
    lang = get_language(code)
    return lang.name if lang else code


def normalize_language_code(code: str) -> str:
    """Convert ISO 639-1 code to ISO 639-3 if possible.
    
    Args:
        code: Language code (ISO 639-1 or ISO 639-3).
        
    Returns:
        ISO 639-3 code if conversion possible, otherwise original code.
    """
    if code in _ISO639_1_TO_3:
        return _ISO639_1_TO_3[code]
    return code


def is_valid_language_code(code: str) -> bool:
    """Check if a language code is valid (known ISO 639-3 or ISO 639-1).
    
    Args:
        code: Language code to validate.
        
    Returns:
        True if the code is recognized.
    """
    return code in LANGUAGES or code in _ISO639_1_TO_3


def get_language_choices_for_gradio(codes: list[str]) -> list[tuple[str, str]]:
    """Get language choices formatted for Gradio dropdown.
    
    Args:
        codes: List of ISO 639-3 language codes.
        
    Returns:
        List of (display_name, code) tuples for Gradio.
    """
    choices = []
    for code in codes:
        lang = get_language(code)
        if lang:
            display = f"{lang.name}"
            if lang.native_name and lang.native_name != lang.name:
                display = f"{lang.name} ({lang.native_name})"
            choices.append((display, code))
        else:
            choices.append((code, code))
    return choices
