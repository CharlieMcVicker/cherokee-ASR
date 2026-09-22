# -*- coding: utf-8 -*-
"""
orthography.py

Backwards compatibility facade re-exporting Cherokee orthography from transcription.cherokee.orthography.
"""

from transcription.cherokee.orthography import (
    GLOTTAL_VARIANTS_REGEX,
    PUNCTUATION_REGEX,
    VOWELS,
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    strip_tones_and_colons,
)

__all__ = [
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "strip_tones_and_colons",
    "PUNCTUATION_REGEX",
    "GLOTTAL_VARIANTS_REGEX",
    "VOWELS",
]
