# -*- coding: utf-8 -*-
"""
transcription.cherokee.orthography

Cherokee orthography models, converters, syllabary maps, and tone normalizers.
"""

from transcription.cherokee.orthography.orthography import (
    GLOTTAL_VARIANTS_REGEX,
    PUNCTUATION_REGEX,
    VOWELS,
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    strip_tones_and_colons,
)
from transcription.cherokee.orthography.syllabary_map import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    phonetics_to_syllabary,
    syllabary_to_phonetics,
)
from transcription.cherokee.orthography.tones import (
    DROPPED_MARKS,
    TONE_DICT,
    UNMARKED_VAL,
    remove_tones_and_double_vowels,
    replace_tones,
    respell_consonants,
)

__all__ = [
    # Orthography
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "strip_tones_and_colons",
    "PUNCTUATION_REGEX",
    "GLOTTAL_VARIANTS_REGEX",
    "VOWELS",
    # Syllabary Map
    "CHEROKEE_SYLLABARY_BASE_MAP",
    "CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP",
    "CHEROKEE_SYLLABARY_MAP",
    "PHONETIC_TO_SYLLABARY_MAP",
    "syllabary_to_phonetics",
    "phonetics_to_syllabary",
    # Tones & Consonants
    "respell_consonants",
    "replace_tones",
    "remove_tones_and_double_vowels",
    "TONE_DICT",
    "DROPPED_MARKS",
    "UNMARKED_VAL",
]
