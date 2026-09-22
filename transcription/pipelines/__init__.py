# -*- coding: utf-8 -*-
"""
transcription.pipelines

Domain and task-specific alignment and transcription pipelines.
"""

from transcription.pipelines.dialogue import (
    DialogueAlignmentPipeline,
    align_dialogue,
    align_syllabary_ctc,
    align_syllabary_greedy,
    build_english_word_tier,
    build_phoneme_tier,
    build_speaker_intervals,
    build_syllabary_word_tier,
    build_turn_intervals,
    export_7tier_textgrid,
)
from transcription.pipelines.enrichment import (
    EnrichmentPipeline,
    EnrichmentRecord,
    calculate_cer,
    calculate_relative_improvement,
    enrich_syllabary,
)
from transcription.pipelines.scripture import (
    ScripturePipeline,
    align_chapter,
    load_bible_chunks,
    load_chapter_transcript,
    reconcile_syllabary_asr,
)

__all__ = [
    "DialogueAlignmentPipeline",
    "align_dialogue",
    "align_syllabary_ctc",
    "align_syllabary_greedy",
    "build_english_word_tier",
    "build_phoneme_tier",
    "build_speaker_intervals",
    "build_syllabary_word_tier",
    "build_turn_intervals",
    "export_7tier_textgrid",
    "EnrichmentPipeline",
    "EnrichmentRecord",
    "calculate_cer",
    "calculate_relative_improvement",
    "enrich_syllabary",
    "ScripturePipeline",
    "align_chapter",
    "reconcile_syllabary_asr",
    "load_chapter_transcript",
    "load_bible_chunks",
]
