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
    load_interview_transcript,
    load_syllabary_transcript,
    prepare_alignment_input,
)
from transcription.alignment.pipeline import (
    align_syllabary_ctc,
    align_syllabary_greedy,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.alignment.reconciliation import (
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_word_intervals,
)
from transcription.alignment.phonotactics import (
    PhonemeCategory,
    PhonotacticAnalysis,
    PhonotacticToken,
    analyze_phonotactics,
    get_intrusion_site_mask,
    get_syncope_mask,
    is_valid_phonotactic_sequence,
    prepare_cherokee_text,
    tokenize_phonemes,
)

__all__ = [
    # Models
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "CTCAlignerConfig",
    "TextChunk",
    "TokenEmission",
    "WordInterval",
    # Ingestion & Normalizers
    "load_bible_chunks",
    "load_generic_chunks",
    "load_interview_transcript",
    "load_syllabary_transcript",
    "prepare_alignment_input",
    "normalize_syllabary_for_alignment",
    "normalize_phonetics_for_alignment",
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
    # Aligners & Runners
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    "CTCSegmentationAligner",
    "get_logits_cached",
    "align_syllabary_greedy",
    "align_syllabary_ctc",
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
    # Phonotactics & Custom Text Preparation
    "PhonemeCategory",
    "PhonotacticToken",
    "PhonotacticAnalysis",
    "tokenize_phonemes",
    "get_syncope_mask",
    "get_intrusion_site_mask",
    "is_valid_phonotactic_sequence",
    "analyze_phonotactics",
    "prepare_cherokee_text",
]
