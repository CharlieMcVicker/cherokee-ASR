# -*- coding: utf-8 -*-
"""
syllabary_map.py

Backwards compatibility facade re-exporting Cherokee syllabary mapping from transcription.cherokee.orthography.
"""

from transcription.cherokee.orthography import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    phonetics_to_syllabary,
    syllabary_to_phonetics,
)

__all__ = [
    "CHEROKEE_SYLLABARY_BASE_MAP",
    "CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP",
    "CHEROKEE_SYLLABARY_MAP",
    "PHONETIC_TO_SYLLABARY_MAP",
    "syllabary_to_phonetics",
    "phonetics_to_syllabary",
]
