"""
Text normalization utilities for alignment.
"""

import re
from transcription.utils.tone_normalization import respell_consonants

PUNCTUATION_REGEX = r"[\,\?\.\!\-\;\:\"\'\“\%\”\(\)\[\]\{\}«»…\’\‘\ʼ\ʻ\`\´\‛]"


def normalize_syllabary_for_alignment(text: str) -> str:
    """
    Normalizes transliterated Cherokee Syllabary text for ASR alignment matching:
    1. Lowercase text and strip hyphens (e.g., A-da-le-ni-s-gv -> adalenisgv).
    2. Convert 'qu' to 'gw'.
    3. Apply consonant & aspiration respelling (t->th, d->t, k->kh, g->k, etc.).
    4. Strip all /h/ sound markers (Syllabary transliteration does not encode aspiration contrast).
    5. Strip numeric tone digits (0-9).
    6. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.replace("-", "")
    text = text.replace("qu", "gw")
    text = respell_consonants(text)
    text = text.replace("h", "")
    text = re.sub(r"\d+", "", text)
    text = re.sub(PUNCTUATION_REGEX, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phonetics_for_alignment(text: str) -> str:
    """
    Normalizes phonetic Cherokee text for ASR alignment matching, preserving aspiration 'h':
    1. Lowercase text and strip hyphens.
    2. Convert 'qu' to 'gw'.
    3. Apply consonant & aspiration respelling (t->th, d->t, k->kh, g->k, etc.).
    4. Preserve /h/ aspiration markers.
    5. Strip numeric tone digits (0-9).
    6. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.replace("-", "")
    text = text.replace("qu", "gw")
    text = respell_consonants(text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(PUNCTUATION_REGEX, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Backward compatibility alias
normalize_text_for_alignment = normalize_syllabary_for_alignment

__all__ = [
    "normalize_syllabary_for_alignment",
    "normalize_phonetics_for_alignment",
    "normalize_text_for_alignment",
]
