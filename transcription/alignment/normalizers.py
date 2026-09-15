"""
Text normalization utilities for alignment.
"""

from transcription.utils.orthography import (
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    strip_tones_and_colons,
)

PUNCTUATION_REGEX = r"[\,\?\.\!\-\;\:\"\“\%\”\(\)\[\]\{\}«»…\´\‛]"
GLOTTAL_VARIANTS_REGEX = r"[\’\‘\ʼ\ʻ\`]"


def normalize_phonetics_for_alignment(
    text: str,
    source: Orthography = Orthography.DG,
) -> str:
    """
    Normalizes phonetic Cherokee text for ASR alignment matching.

    Default source is Orthography.DG (transliteration / dictionary phonetics).
    Target is canonical Orthography.TTH (alignment and ASR acoustic representation).
    When source is Orthography.TTH, conversion is an exact no-op on consonants.
    """
    if not text:
        return ""

    return convert_orthography(text, source=source, target=Orthography.TTH)


def normalize_syllabary_for_alignment(
    text: str,
    source: Orthography = Orthography.SYLLABARY,
    target: Orthography = Orthography.TTH,
) -> str:
    """
    Normalizes Cherokee Syllabary (or Latin transliteration) for ASR alignment matching.

    Default maps Cherokee Syllabary characters to canonical Orthography.TTH alignment phonetics.
    """
    if not text:
        return ""

    return convert_orthography(text, source=source, target=target)


# Backward compatibility alias
normalize_text_for_alignment = normalize_phonetics_for_alignment

__all__ = [
    "Orthography",
    "convert_orthography",
    "normalize_phonetics_for_alignment",
    "normalize_syllabary_for_alignment",
    "normalize_text_for_alignment",
]
