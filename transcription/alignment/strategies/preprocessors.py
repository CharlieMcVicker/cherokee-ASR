"""
Phonetic and syllabary preprocessor strategy implementations.
"""

from typing import Optional
from transcription.alignment.ports.protocols import PhoneticPreprocessor
from transcription.utils.syllabary_map import CHEROKEE_SYLLABARY_MAP


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
        from transcription.timestamping.prepare_ground_truth import (
            normalize_text_for_alignment,
        )

        return normalize_text_for_alignment(text)


class SyllabaryToPhoneticPreprocessor:
    """
    Converts Cherokee syllabary characters to phonetic base using CHEROKEE_SYLLABARY_MAP,
    then applies the injected PhoneticPreprocessor (or CherokeePhoneticPreprocessor by default).
    """

    def __init__(self, phonetic_cleaner: Optional[PhoneticPreprocessor] = None):
        self.cleaner = (
            phonetic_cleaner
            if phonetic_cleaner is not None
            else CherokeePhoneticPreprocessor()
        )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Convert syllabary characters to base phonetic representations
        converted = "".join(CHEROKEE_SYLLABARY_MAP.get(ch, ch) for ch in text)
        return self.cleaner.normalize(converted)
