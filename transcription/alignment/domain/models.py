"""
Core domain models for transcription and timestamping alignment.

Pure dataclasses with generic chunk semantics, decoupled from dataset-specific
abstractions (such as Bible verses or chapter numbers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TokenEmission:
    """An individual token or word emitted by the ASR model with timestamp bounds."""

    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass
class TextChunk:
    """Generic text unit to be aligned against audio/emissions."""

    chunk_id: str
    raw_text: str
    normalized_text: str = ""
    syllabary_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WordInterval:
    """Aligned word token with start/end bounds and reconciliation details."""

    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    flagged: bool = False
    syllabary: Optional[str] = None
    reconciled_word: Optional[str] = None
    emitted_word: Optional[str] = None


@dataclass
class AlignedChunk:
    """A matched segment with bounded timestamps and aligned words."""

    chunk_id: str
    chunk: TextChunk
    start_sec: float
    end_sec: float
    words: List[WordInterval] = field(default_factory=list)
    distance_score: float = 1.0
    emitted_text: str = ""


@dataclass
class AlignmentMetrics:
    """Quality metrics across aligned chunks."""

    total_chunks: int
    matched_chunks: int
    match_ratio: float
    mean_distance_score: float
    total_ground_truth_chars: int
    total_emitted_chars: int


@dataclass
class AlignmentOutput:
    """Final domain output from alignment execution."""

    aligned_chunks: List[AlignedChunk]
    source_id: str = ""
    raw_tokens: List[TokenEmission] = field(default_factory=list)
    metrics: Optional[AlignmentMetrics] = None
