# -*- coding: utf-8 -*-
"""
transcription.cherokee

Domain-specific Cherokee phonetic, linguistic, orthographic representations, and models.
"""

from transcription.cherokee.models import (
    CherokeeASRModel,
    load_cherokee_asr_model,
)
from transcription.cherokee.orthography import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    phonetics_to_syllabary,
    remove_tones_and_double_vowels,
    replace_tones,
    respell_consonants,
    strip_tones_and_colons,
    syllabary_to_phonetics,
)

__all__ = [
    # Orthography
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "strip_tones_and_colons",
    # Syllabary Map
    "CHEROKEE_SYLLABARY_BASE_MAP",
    "CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP",
    "CHEROKEE_SYLLABARY_MAP",
    "PHONETIC_TO_SYLLABARY_MAP",
    "syllabary_to_phonetics",
    "phonetics_to_syllabary",
    # Tones
    "respell_consonants",
    "replace_tones",
    "remove_tones_and_double_vowels",
    # Models
    "CherokeeASRModel",
    "load_cherokee_asr_model",
]
