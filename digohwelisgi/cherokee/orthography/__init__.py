# -*- coding: utf-8 -*-
"""
digohwelisgi.cherokee.orthography

Cherokee orthography models, converters, syllabary maps, and tone normalizers.
"""

from digohwelisgi.cherokee.orthography.orthography import (
    GLOTTAL_VARIANTS_REGEX,
    PUNCTUATION_REGEX,
    VOWELS,
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    strip_tones_and_colons,
)
from digohwelisgi.cherokee.orthography.syllabary_map import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    phonetics_to_syllabary,
    syllabary_to_phonetics,
)
from digohwelisgi.cherokee.orthography.tones import (
    DROPPED_MARKS,
    MORA_TO_TONE_MAP,
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
    TONE_DICT,
    TONE_TO_MORA_MAP,
    UNMARKED_VAL,
    clean_asterisks_with_syllabary,
    collapse_double_vowels,
    decode_cv_ids,
    decode_tone_ids,
    encode_cv_text,
    encode_tone_tokens,
    moraic_tokens_to_pitch_contour,
    normalize_standard2_cv,
    reconcile_dual_stream,
    remove_tones_and_double_vowels,
    replace_tones,
    resolve_vowel_hiatus,
    respell_consonants,
    save_dual_vocabularies,
    standardize_glottal_stops,
    to_canonical_toned_tth,
    to_moraic_tone_tokens,
)

normalize_text = clean_punctuation_and_whitespace

__all__ = [
    # Orthography
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "normalize_text",
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
    "standardize_glottal_stops",
    "resolve_vowel_hiatus",
    "collapse_double_vowels",
    "respell_consonants",
    "replace_tones",
    "remove_tones_and_double_vowels",
    "clean_asterisks_with_syllabary",
    "to_canonical_toned_tth",
    "to_moraic_tone_tokens",
    "reconcile_dual_stream",
    "moraic_tokens_to_pitch_contour",
    "normalize_standard2_cv",
    "encode_cv_text",
    "decode_cv_ids",
    "encode_tone_tokens",
    "decode_tone_ids",
    "save_dual_vocabularies",
    "TONE_DICT",
    "TONE_TO_MORA_MAP",
    "MORA_TO_TONE_MAP",
    "MORAIC_TONE_VOCAB",
    "SEGMENTAL_CV_VOCAB",
    "DROPPED_MARKS",
    "UNMARKED_VAL",
]
