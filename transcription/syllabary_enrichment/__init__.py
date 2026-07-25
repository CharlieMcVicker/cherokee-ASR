# -*- coding: utf-8 -*-
"""
transcription.syllabary_enrichment package
"""

from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
    align_character_syllable_detailed,
    SyllableAlignment,
    is_cherokee_syllable,
    get_base_transliteration,
)
from transcription.syllabary_enrichment.batch_inference_aligner import (
    load_manifest,
    run_batch_inference,
    process_batch_inference_and_alignment,
)

from transcription.syllabary_enrichment.enrich_syllabary import reconcile_phonetics
from transcription.syllabary_enrichment.evaluate_reconciliation import (
    calculate_cer,
    calculate_relative_improvement,
    run_evaluation,
    print_summary_table,
)

__all__ = [
    "align_character_syllable",
    "align_character_syllable_detailed",
    "SyllableAlignment",
    "is_cherokee_syllable",
    "get_base_transliteration",
    "load_manifest",
    "run_batch_inference",
    "process_batch_inference_and_alignment",
    "reconcile_phonetics",
    "calculate_cer",
    "calculate_relative_improvement",
    "run_evaluation",
    "print_summary_table",
]
