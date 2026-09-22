# -*- coding: utf-8 -*-
"""
pipeline.py

High-level programmatic alignment runners for Cherokee audio and transcripts.
Provides unified, turnkey functions for:
1. Greedy ASR inference + DTW + Syllabary Reconciliation (align_syllabary_greedy)
2. Syncope- and intrusion-aware CTC Segmentation (align_syllabary_ctc)

Forwards to canonical pipeline implementations in transcription.pipelines.dialogue.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from transcription.pipelines.dialogue.pipeline import DialogueAlignmentPipeline
else:

    class DialogueAlignmentPipeline:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            from transcription.pipelines.dialogue.pipeline import (
                DialogueAlignmentPipeline as _DAP,
            )

            return _DAP(*args, **kwargs)


def align_dialogue(*args: Any, **kwargs: Any) -> Any:
    """Run full dialogue alignment pipeline."""
    from transcription.pipelines.dialogue.pipeline import align_dialogue as _fn

    return _fn(*args, **kwargs)


def align_syllabary_ctc(*args: Any, **kwargs: Any) -> Any:
    """Run CTC segmentation alignment with syncope- and intrusion-aware Cherokee phonotactics."""
    from transcription.pipelines.dialogue.pipeline import align_syllabary_ctc as _fn

    return _fn(*args, **kwargs)


def align_syllabary_greedy(*args: Any, **kwargs: Any) -> Any:
    """Run greedy ASR inference + DTW + syllabary phonetic reconciliation."""
    from transcription.pipelines.dialogue.pipeline import align_syllabary_greedy as _fn

    return _fn(*args, **kwargs)


def build_syllabary_word_tier(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import (
        build_syllabary_word_tier as _fn,
    )

    return _fn(*args, **kwargs)


def build_english_word_tier(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import build_english_word_tier as _fn

    return _fn(*args, **kwargs)


_build_syllabary_word_tier = build_syllabary_word_tier
_build_english_word_tier = build_english_word_tier


def build_speaker_intervals(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import build_speaker_intervals as _fn

    return _fn(*args, **kwargs)


def build_turn_intervals(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import build_turn_intervals as _fn

    return _fn(*args, **kwargs)


def build_phoneme_tier(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import build_phoneme_tier as _fn

    return _fn(*args, **kwargs)


def export_7tier_textgrid(*args: Any, **kwargs: Any) -> Any:
    from transcription.pipelines.dialogue.pipeline import export_7tier_textgrid as _fn

    return _fn(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "DialogueAlignmentPipeline":
        from transcription.pipelines.dialogue.pipeline import DialogueAlignmentPipeline

        return DialogueAlignmentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DialogueAlignmentPipeline",
    "align_dialogue",
    "align_syllabary_greedy",
    "align_syllabary_ctc",
    "_build_syllabary_word_tier",
    "_build_english_word_tier",
    "build_syllabary_word_tier",
    "build_english_word_tier",
    "build_speaker_intervals",
    "build_turn_intervals",
    "build_phoneme_tier",
    "export_7tier_textgrid",
]
