"""
Phonetic and syllabary preprocessor strategy implementations.
"""

import re
from typing import Optional
from transcription.alignment.ports.protocols import PhoneticPreprocessor
from transcription.utils.syllabary_map import CHEROKEE_SYLLABARY_MAP
from transcription.utils.tone_normalization import respell_consonants


def normalize_text_for_alignment(text: str) -> str:
    """
    Normalizes transliterated Cherokee text for ASR alignment matching:
    1. Lowercase text and strip hyphens (e.g., A-da-le-ni-s-gv -> adalenisgv).
    2. Convert 'qu' to 'gw'.
    3. Apply consonant & aspiration respelling (t->th, d->t, k->kh, g->k, etc.).
    4. Strip /h/ sound markers.
    5. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    # Strip hyphens
    text = text.replace("-", "")

    # Convert qu -> gw
    text = text.replace("qu", "gw")

    # Respell consonants
    text = respell_consonants(text)

    # Drop all /h/ sound markers
    text = text.replace("h", "")

    # Strip punctuation
    punctuation_regex = r"[\,\?\.\!\-\;\:\"\'\“\%\”\(\)\[\]\{\}«»…\’\‘\ʼ\ʻ\`\´\‛]"
    text = re.sub(punctuation_regex, "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CherokeePhoneticPreprocessor:
    """
    Normalizes transliterated Cherokee phonetic text:
    - Strips hyphens
    - Converts 'qu' -> 'gw'
    - Applies consonant and aspiration respelling
    - Strips /h/ sound markers
    - Removes punctuation and collapses whitespace
    """

    def normalize(self, text: str) -> str:
        return normalize_text_for_alignment(text)


class SyllabaryToPhoneticPreprocessor:
    """
    Converts Cherokee syllabary characters to phonetic base using CHEROKEE_SYLLABARY_MAP,
    then applies the injected PhoneticPreprocessor.
    """

    def __init__(self, phonetic_cleaner: Optional[PhoneticPreprocessor] = None):
        self.cleaner = phonetic_cleaner or CherokeePhoneticPreprocessor()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Convert syllabary characters to base phonetic representations
        converted = "".join(CHEROKEE_SYLLABARY_MAP.get(ch, ch) for ch in text)
        return self.cleaner.normalize(converted)
