"""
transcription.alignment package.

Streamlined functional architecture for audio and text chunk alignment.
"""

from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.ctc_aligner import (
    CTCSegmentationAligner,
    get_logits_cached,
)
from transcription.alignment.cli import run_alignment_pipeline
from transcription.alignment.calibrated_distance_metrics import (
    PhonologicalConfusionCostMetric,
)
from transcription.alignment.distance_metrics import (
    CharacterErrorRateMetric,
    ConfusionMatrixCostMetric,
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
    CachedASREmissionsExtractor,
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
    prepare_audio_chunks,
)
from transcription.alignment.ingestion import (
    load_bible_chunks,
    load_generic_chunks,
    prepare_alignment_input,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
    normalize_text_for_alignment,
)
from transcription.alignment.reconciliation import (
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_word_intervals,
)
from transcription.alignment.threshold_finder import (
    AlignmentRecord,
    AlignmentThresholdFinder,
    ThresholdMetrics,
    ThresholdSearchStep,
    find_threshold_bounds,
    load_alignment_records,
    parse_verse_reference,
)

__all__ = [
    # Models
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "TextChunk",
    "TokenEmission",
    "WordInterval",
    # Ingestion & Normalizers
    "load_bible_chunks",
    "load_generic_chunks",
    "prepare_alignment_input",
    "normalize_syllabary_for_alignment",
    "normalize_phonetics_for_alignment",
    "normalize_text_for_alignment",
    # Extractors
    "ASREmissionsExtractor",
    "CachedASREmissionsExtractor",
    "CherokeeASRExtractor",
    "CallbackEmissionsExtractor",
    "PrecomputedEmissionsExtractor",
    "prepare_audio_chunks",
    # Distance Metrics
    "DistanceMetric",
    "DefaultCERDistanceMetric",
    "CharacterErrorRateMetric",
    "ConfusionMatrixCostMetric",
    "PhonologicalConfusionCostMetric",
    "LevenshteinDistanceMetric",
    "CustomCallableDistanceMetric",
    "calculate_cer",
    # Aligners
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    "CTCSegmentationAligner",
    "get_logits_cached",
    # Reconciliation
    "reconcile_word_intervals",
    "reconcile_alignment_words",
    "reconcile_alignment_by_chunk",
    # Exporters
    "export_textgrid",
    "export_manifest",
    "export_debug_json",
    # Pipeline / CLI
    "run_alignment_pipeline",
    # Threshold Finder
    "AlignmentRecord",
    "AlignmentThresholdFinder",
    "ThresholdMetrics",
    "ThresholdSearchStep",
    "find_threshold_bounds",
    "load_alignment_records",
    "parse_verse_reference",
]
