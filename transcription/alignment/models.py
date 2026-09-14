"""
Core domain models for transcription and timestamping alignment.

Pure dataclasses with generic chunk semantics.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
