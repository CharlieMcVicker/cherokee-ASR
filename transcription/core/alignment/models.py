# -*- coding: utf-8 -*-
"""
models.py

Pure domain models for alignment, dynamic programming, and CTC segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class CTCAlignerConfig:
    """Strongly-typed language-agnostic configuration for CTC segmentation alignment."""

    syncope_tokens: Tuple[Tuple[str, ...] | str, ...] = ()
    intrusive_tokens: Tuple[str, ...] = ()
    intrusive_max_stride: int = 0
    flag_min_confidence: float = 0.05
    flag_min_char_confidence: float = 0.0
    index_duration: float = 0.02
    min_window_size: int = 8000
    max_window_size: int = 100000
    buffer_trail_ms: int = 300
    buffer_lead_ms: int = 100
    boundary_pad_sec: float = 0.1
    chunk_seconds: float = 30.0
    margin_seconds: float = 1.0
    cache: bool = True
    cache_dir: Optional[Path] = None
    enable_vad_soft_masking: bool = False
    vad_p_low: float = 0.15
    vad_p_high: float = 0.60
    vad_pad_ms: int = 60


@dataclass(frozen=True)
class TokenEmission:
    """An individual token or word emitted by an ASR model with timestamp bounds."""

    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass(frozen=True)
class TextChunk:
    """Generic text unit to be aligned against audio/emissions."""

    chunk_id: str
    text: str


@dataclass
class WordInterval:
    """Aligned word token with start/end bounds and emitted token details."""

    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    flagged: bool = False
    emitted_word: Optional[str] = None
    min_char_confidence: Optional[float] = None


@dataclass
class AlignedChunk:
    """A matched segment with bounded timestamps and aligned words."""

    chunk_id: str
    start_sec: float
    end_sec: float
    words: List[WordInterval] = field(default_factory=list)
    distance_score: float = 1.0
    emitted_text: str = ""

    @property
    def has_anomalies(self) -> bool:
        return any(w.flagged for w in self.words)

    @property
    def flagged_words(self) -> List[WordInterval]:
        return [w for w in self.words if w.flagged]


@dataclass
class AlignmentMetrics:
    """Quality metrics across aligned chunks."""

    total_chunks: int
    matched_chunks: int
    match_ratio: float
    mean_distance_score: float
    total_ground_truth_chars: int
    total_emitted_chars: int
    flagged_words_count: int = 0


@dataclass
class AlignmentOutput:
    """Final output from alignment execution."""

    aligned_chunks: List[AlignedChunk]
    source_id: str = ""
    raw_tokens: List[TokenEmission] = field(default_factory=list)
    metrics: Optional[AlignmentMetrics] = None


__all__ = [
    "CTCAlignerConfig",
    "TokenEmission",
    "TextChunk",
    "WordInterval",
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
]
