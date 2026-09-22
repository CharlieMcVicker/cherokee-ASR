#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tone_normalization.py

Backwards compatibility facade re-exporting tone normalization from transcription.cherokee.orthography.
"""

from transcription.cherokee.orthography import (
    DROPPED_MARKS,
    TONE_DICT,
    UNMARKED_VAL,
    VOWELS,
    remove_tones_and_double_vowels,
    replace_tones,
    respell_consonants,
    strip_tones_and_colons,
)

__all__ = [
    "respell_consonants",
    "strip_tones_and_colons",
    "replace_tones",
    "remove_tones_and_double_vowels",
    "TONE_DICT",
    "DROPPED_MARKS",
    "UNMARKED_VAL",
    "VOWELS",
]
