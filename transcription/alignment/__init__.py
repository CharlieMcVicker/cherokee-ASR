"""
transcription.alignment package.

Streamlined functional architecture for audio and text chunk alignment.
"""

from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.cli import run_alignment_pipeline
from transcription.alignment.distance_metrics import (
    CharacterErrorRateMetric,
    CustomCallableDistanceMetric,
    DefaultCERDistanceMetric,
    DistanceMetric,
    LevenshteinDistanceMetric,
    calculate_cer,
)
from transcription.alignment.exporters import (
    export_debug_json,
    export_manifest,
    export_textgrid,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
    prepare_audio_chunks,
)
from transcription.alignment.ingestion import (
    load_bible_chunks,
    load_generic_chunks,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import normalize_text_for_alignment
from transcription.alignment.reconciliation import reconcile_alignment

__all__ = [
    # Models
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "TextChunk",
    "TokenEmission",
    "WordInterval",
    # Ingestion
    "load_bible_chunks",
    "load_generic_chunks",
    # Normalizers
    "normalize_text_for_alignment",
    # Extractors
    "ASREmissionsExtractor",
    "CherokeeASRExtractor",
    "CallbackEmissionsExtractor",
    "PrecomputedEmissionsExtractor",
    "prepare_audio_chunks",
    # Distance Metrics
    "DistanceMetric",
    "DefaultCERDistanceMetric",
    "CharacterErrorRateMetric",
    "LevenshteinDistanceMetric",
    "CustomCallableDistanceMetric",
    "calculate_cer",
    # Aligners
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    # Reconciliation
    "reconcile_alignment",
    # Exporters
    "export_textgrid",
    "export_manifest",
    "export_debug_json",
    # Pipeline / CLI
    "run_alignment_pipeline",
]
