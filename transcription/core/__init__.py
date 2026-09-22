# Core language-agnostic components
from transcription.core.models.output import ModelOutput
from transcription.core.models.model import ASRModel
from transcription.core.models.inference import (
    infer_emissions,
    infer_emissions_batch,
    preprocess_audio,
)
from transcription.core.exporters import (
    IntervalTier,
    TextGridBuilder,
    export_textgrid,
    export_manifest,
    export_debug_json,
)
from transcription.core.alignment import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    CTCSegmentationAligner,
    DefaultCERDistanceMetric,
    DistanceMetric,
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
    TextChunk,
    TextPreparerProtocol,
    TokenEmission,
    WordInterval,
)

__all__ = [
    "ModelOutput",
    "ASRModel",
    "infer_emissions",
    "infer_emissions_batch",
    "preprocess_audio",
    "IntervalTier",
    "TextGridBuilder",
    "export_textgrid",
    "export_manifest",
    "export_debug_json",
    "CTCAlignerConfig",
    "TokenEmission",
    "TextChunk",
    "WordInterval",
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "DistanceMetric",
    "DefaultCERDistanceMetric",
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    "CTCSegmentationAligner",
    "TextPreparerProtocol",
]
