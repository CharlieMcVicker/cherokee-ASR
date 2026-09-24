# -*- coding: utf-8 -*-
"""
transcription.cherokee.phonotactics

Cherokee surface phonotactic constraints, token classification,
syncope/intrusion masks, and CTC segmentation text preparation.
"""

from transcription.cherokee.phonotactics.phonotactics import (
    ASPIRATED_STOP_SET,
    CHEROKEE_DEFAULT_INTRUSIVE_MAX_STRIDE,
    CHEROKEE_DEFAULT_INTRUSIVE_TOKENS,
    CHEROKEE_DEFAULT_SYNCOPE_TOKENS,
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
    create_cherokee_ctc_config,
    get_intrusion_site_mask,
    get_syncope_mask,
    is_valid_phonotactic_sequence,
    prepare_cherokee_direct,
    prepare_cherokee_text,
    prepare_cherokee_with_intrusion,
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
    "CHEROKEE_DEFAULT_SYNCOPE_TOKENS",
    "CHEROKEE_DEFAULT_INTRUSIVE_TOKENS",
    "CHEROKEE_DEFAULT_INTRUSIVE_MAX_STRIDE",
    "create_cherokee_ctc_config",
    "prepare_cherokee_text",
    "prepare_cherokee_with_intrusion",
    "prepare_cherokee_direct",
]
