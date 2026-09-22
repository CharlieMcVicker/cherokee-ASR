# -*- coding: utf-8 -*-
"""
transcription.pipelines.dialogue module.

Multi-speaker code-switched dialogue alignment pipeline orchestrating
token discrimination, acoustic CTC segmentation, phonetic syllabary reconciliation,
and 7-tier Praat TextGrid and JSON manifest exporting.
"""

from transcription.pipelines.dialogue.pipeline import (
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
]
