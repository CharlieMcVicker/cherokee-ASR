# -*- coding: utf-8 -*-
"""
transcription.syllabary_enrichment.enrich_syllabary module.

Compatibility shim forwarding to transcription.cherokee.enrichment.syllable_alignment.
"""

from transcription.cherokee.enrichment.syllable_alignment import (
    _enrich_single_syllable,
    _get_base_syllable,
    reconcile_phonetics,
)

__all__ = [
    "reconcile_phonetics",
    "_enrich_single_syllable",
    "_get_base_syllable",
]
