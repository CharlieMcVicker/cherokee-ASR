# -*- coding: utf-8 -*-
"""
phonotactics.py

Cherokee Phonotactic Engine and Rule Parser.
Re-exported from transcription.cherokee.phonotactics for backwards compatibility.
"""

from transcription.cherokee.phonotactics import (
    ASPIRATED_STOP_SET,
    LARYNGEAL_SET,
    PLAIN_SONORANT_SET,
    PLAIN_STOP_SET,
    SIBILANT_CLUSTER_SET,
    SIBILANT_SET,
    VOICELESS_SONORANT_SET,
    VOWEL_SET,
    PhonemeCategory,
    PhonotacticAnalysis,
    PhonotacticToken,
    analyze_phonotactics,
    get_intrusion_site_mask,
    get_syncope_mask,
    is_valid_phonotactic_sequence,
    prepare_cherokee_text,
    tokenize_phonemes,
)

__all__ = [
    "PhonemeCategory",
    "PhonotacticToken",
    "PhonotacticAnalysis",
    "VOWEL_SET",
    "PLAIN_STOP_SET",
    "ASPIRATED_STOP_SET",
    "SIBILANT_SET",
    "PLAIN_SONORANT_SET",
    "VOICELESS_SONORANT_SET",
    "LARYNGEAL_SET",
    "SIBILANT_CLUSTER_SET",
    "tokenize_phonemes",
    "get_syncope_mask",
    "get_intrusion_site_mask",
    "is_valid_phonotactic_sequence",
    "analyze_phonotactics",
    "prepare_cherokee_text",
]
