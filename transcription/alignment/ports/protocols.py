"""
Port definitions (Protocols) for alignment engine components.
"""

from typing import Any, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from transcription.alignment.domain.models import (
    AlignmentOutput,
    TextChunk,
    TokenEmission,
)


@runtime_checkable
class PhoneticPreprocessor(Protocol):
    """Port for normalizing text/syllabary into comparable phonetics."""

    def normalize(self, text: str) -> str:
        """Transforms input text into standardized phonetic string."""
        ...


@runtime_checkable
class DistanceMetric(Protocol):
    """Port for computing edit cost between hypothesis and reference."""

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """Returns distance/cost metric (e.g. CER in [0.0, inf])."""
        ...


@runtime_checkable
class ASREmissionsExtractor(Protocol):
    """Port for generating token emissions from audio."""

    def extract(self, audio_input: Any) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input."""
        ...


@runtime_checkable
class ReconciliationStrategy(Protocol):
    """Port for enriching transliteration with acoustic ASR tokens."""

    def reconcile(
        self, syllabary_text: str, emitted_text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Reconciles canonical syllabary transliteration with emitted ASR text."""
        ...


@runtime_checkable
class ChunkAlignmentEngine(Protocol):
    """Core domain alignment port."""

    def align_chunks(
        self,
        emissions: Sequence[TokenEmission],
        chunks: Sequence[TextChunk],
    ) -> AlignmentOutput:
        """Aligns text chunks against sequence of token emissions."""
        ...
