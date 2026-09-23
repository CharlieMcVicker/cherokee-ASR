# -*- coding: utf-8 -*-
"""
transcription.pipelines.enrichment package.

Domain pipeline for Cherokee syllabary phonetic enrichment, fine-grained syllable
alignment, and acoustic reconciliation against ASR emissions.
"""

from transcription.pipelines.enrichment.pipeline import (
    EnrichmentPipeline,
    EnrichmentRecord,
    calculate_cer,
    calculate_relative_improvement,
    enrich_syllabary,
)

__all__ = [
    "EnrichmentPipeline",
    "EnrichmentRecord",
    "calculate_cer",
    "calculate_relative_improvement",
    "enrich_syllabary",
]
