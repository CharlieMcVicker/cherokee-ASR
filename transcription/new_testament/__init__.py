"""
New Testament audio and transcript matching & syllabary/ASR reconciliation package.
Delegates to transcription.pipelines.scripture for canonical pipeline implementation.
"""

from transcription.pipelines.scripture import (
    align_chapter,
    load_chapter_transcript,
    reconcile_syllabary_asr,
)

__all__ = [
    "load_chapter_transcript",
    "align_chapter",
    "reconcile_syllabary_asr",
]
