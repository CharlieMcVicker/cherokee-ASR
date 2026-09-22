"""
New Testament Pipeline Module for Audio-Transcript Alignment and Syllabary/ASR Reconciliation.
Delegates to transcription.pipelines.scripture.pipeline.
"""

import sys
from typing import Any, Optional, Union
from pathlib import Path

from transcription.pipelines.scripture.pipeline import (
    ScripturePipeline,
    align_chapter as _scripture_align_chapter,
    reconcile_syllabary_asr,
    CTCSegmentationAligner,
)
from transcription.pipelines.scripture.ingestion import load_chapter_transcript
from transcription.core.alignment.models import AlignmentOutput, CTCAlignerConfig
from transcription.cherokee.models import CherokeeASRModel


from transcription.alignment.ctc_aligner import (
    CTCSegmentationAligner as BaseCTCSegmentationAligner,
)


def align_chapter(
    audio_path: Union[str, Path],
    transcript_path: Union[str, Path],
    output_dir: Union[str, Path] = "output_praat/new_testament",
    export_praat: bool = True,
    export_manifest: bool = True,
    debug_export: bool = False,
    model_path: Optional[str] = None,
    skip_vad: bool = False,
    reconcile: bool = True,
    distance_metric: Any = None,
    emissions_extractor: Any = None,
    model_revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    engine: str = "ctc",
    ctc_aligner: Optional[Any] = None,
    asr_model: Optional[CherokeeASRModel] = None,
    cache: bool = True,
    aligner_config: Optional[CTCAlignerConfig] = None,
) -> AlignmentOutput:
    """Delegates to transcription.pipelines.scripture.pipeline.align_chapter, honoring module-level patches."""
    current_module = sys.modules.get(__name__)
    current_aligner_cls = getattr(
        current_module, "CTCSegmentationAligner", BaseCTCSegmentationAligner
    )
    if ctc_aligner is None and current_aligner_cls is not BaseCTCSegmentationAligner:
        # Patch active on transcription.new_testament.pipeline.CTCSegmentationAligner
        ctc_aligner = current_aligner_cls(
            model=asr_model,
            config=aligner_config or CTCAlignerConfig(),
        )

    return _scripture_align_chapter(
        audio_path=audio_path,
        transcript_path=transcript_path,
        output_dir=output_dir,
        export_praat=export_praat,
        export_manifest=export_manifest,
        debug_export=debug_export,
        model_path=model_path,
        skip_vad=skip_vad,
        reconcile=reconcile,
        distance_metric=distance_metric,
        emissions_extractor=emissions_extractor,
        model_revision=model_revision,
        cache_dir=cache_dir,
        engine=engine,
        ctc_aligner=ctc_aligner,
        asr_model=asr_model,
        cache=cache,
        aligner_config=aligner_config,
    )


__all__ = [
    "ScripturePipeline",
    "load_chapter_transcript",
    "align_chapter",
    "reconcile_syllabary_asr",
    "CTCSegmentationAligner",
]
