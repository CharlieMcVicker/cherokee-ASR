"""
Core domain models for transcription and timestamping alignment.

Pure dataclasses with generic chunk semantics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from transcription.utils.orthography import Orthography


@dataclass(frozen=True)
class CTCAlignerConfig:
    """Strongly-typed configuration for syncope- and intrusion-aware CTC alignment."""

    syncope_tokens: Tuple[str, ...] = ("a", "e", "i", "o", "u", "v", "t")
    intrusive_tokens: Tuple[str, ...] = ("h", "'")
    intrusive_max_stride: int = 4
    enforce_phonotactics: bool = True
    flag_min_confidence: float = 0.01
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


@dataclass(frozen=True)
class TokenEmission:
    """An individual token or word emitted by the ASR model with timestamp bounds."""

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
