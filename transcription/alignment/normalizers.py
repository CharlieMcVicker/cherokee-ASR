"""
Text normalization utilities for alignment.
"""

import re
from transcription.utils.syllabary_map import cherokee_to_bad_phonetics
from transcription.utils.tone_normalization import respell_consonants

PUNCTUATION_REGEX = r"[\,\?\.\!\-\;\:\"\“\%\”\(\)\[\]\{\}«»…\´\‛]"
GLOTTAL_VARIANTS_REGEX = r"[\’\‘\ʼ\ʻ\`]"


def normalize_phonetics_for_alignment(text: str) -> str:
    """
    Normalizes phonetic Cherokee text for ASR alignment matching, preserving aspiration 'h':
    1. Lowercase text and strip hyphens.
    2. Convert 'qu' to 'gw'.
    3. Map 'tl' -> 'dl' so citation 'tl' (Ꮯ, etc.) defaults to unaspirated 'tl' via respell_consonants.
    4. Apply consonant & aspiration respelling (t->th, d->t, k->kh, g->k, etc.).
    5. Strip numeric tone digits (0-9).
    6. Normalize glottal variants to apostrophe (').
    7. Insert hiatus glottal stop /'/ between adjacent vowels (e.g., e-hna-i -> enha'i).
    8. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.replace("-", "")
    text = text.replace("qu", "gw")
    text = text.replace("tl", "dl")
    text = respell_consonants(text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(GLOTTAL_VARIANTS_REGEX, "'", text)
    # Insert required hiatus glottal stop between adjacent vowels
    text = re.sub(r"([aeiouvAEIOUV])(?=[aeiouvAEIOUV])", r"\1'", text)
    text = re.sub(PUNCTUATION_REGEX, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_syllabary_for_alignment(text: str) -> str:
    """
    Normalizes Cherokee Syllabary (or transliterated text) for ASR alignment matching:
    1. If Cherokee Syllabary characters are present, maps them via syllabary_map (cherokee_to_bad_phonetics).
    2. If Latin transliteration, applies tl->dl and consonant respelling.
    3. Lowercase text and strip hyphens.
    4. Convert 'qu' to 'gw'.
    5. Strip numeric tone digits (0-9).
    6. Normalize glottal variants to apostrophe (').
    7. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    has_cherokee = any(
        0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF for c in text
    )
    if has_cherokee:
        text = cherokee_to_bad_phonetics(text.upper())
    else:
        text = text.replace("tl", "dl").replace("TL", "dl")
        text = respell_consonants(text)

    text = text.lower()
    text = text.replace("-", "")
    text = text.replace("qu", "gw")
    text = re.sub(r"\d+", "", text)
    text = re.sub(GLOTTAL_VARIANTS_REGEX, "'", text)
    text = re.sub(PUNCTUATION_REGEX, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Backward compatibility alias
normalize_text_for_alignment = normalize_phonetics_for_alignment

__all__ = [
    "normalize_phonetics_for_alignment",
    "normalize_syllabary_for_alignment",
    "normalize_text_for_alignment",
]
