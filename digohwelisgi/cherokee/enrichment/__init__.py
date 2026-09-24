# -*- coding: utf-8 -*-
"""
digohwelisgi.cherokee.enrichment

Cherokee syllabary enrichment package: fine-grained syllable alignment and phonetic
reconciliation using acoustic ASR emissions while preserving Cherokee Syllabary
as the immutable structural anchor.
"""

from digohwelisgi.cherokee.enrichment.syllable_alignment import (
    SyllableAlignment,
    SyllableAlignmentEngine,
    align_character_syllable,
    align_character_syllable_detailed,
    get_base_transliteration,
    is_cherokee_syllable,
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_phonetics,
    reconcile_word_intervals,
)

__all__ = [
    "SyllableAlignment",
    "SyllableAlignmentEngine",
    "align_character_syllable",
    "align_character_syllable_detailed",
    "get_base_transliteration",
    "is_cherokee_syllable",
    "reconcile_alignment_by_chunk",
    "reconcile_alignment_words",
    "reconcile_phonetics",
    "reconcile_word_intervals",
]
