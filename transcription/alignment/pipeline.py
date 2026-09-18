# -*- coding: utf-8 -*-
"""
pipeline.py

High-level programmatic alignment runners for Cherokee audio and transcripts.
Provides unified, turnkey functions for:
1. Greedy ASR inference + DTW + Syllabary Reconciliation (align_syllabary_greedy)
2. Syncope- and intrusion-aware CTC Segmentation (align_syllabary_ctc)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydub import AudioSegment

from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.exporters import (
    export_debug_json,
    export_manifest as export_manifest_file,
    export_textgrid,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CherokeeASRExtractor,
)
from transcription.alignment.ingestion import (
    load_syllabary_transcript,
    prepare_alignment_input,
)
from transcription.alignment.models import (
    AlignmentOutput,
    CTCAlignerConfig,
    WordInterval,
)
from transcription.alignment.reconciliation import reconcile_alignment_words
from transcription.models.asr_model import CherokeeASRModel


def _build_syllabary_word_tier(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> List[WordInterval]:
    """
    Builds a list of WordIntervals containing the original Cherokee syllabary tokens
    mapped to aligned word timestamps.
    """
    syllabary_words: List[WordInterval] = []
    for chunk in alignment.aligned_chunks:
        syll_chunk_str = syllabary_lookup.get(chunk.chunk_id, "")
        s_words = syll_chunk_str.split()
        for w_idx, w in enumerate(chunk.words):
            syll_w = s_words[w_idx] if w_idx < len(s_words) else w.word
            syllabary_words.append(
                WordInterval(
                    word=syll_w,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=w.emitted_word,
                )
            )
    return syllabary_words


def align_syllabary_greedy(
    audio: Union[str, Path, AudioSegment, np.ndarray],
    transcript: Union[str, Path, List[str], List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[CherokeeASRModel] = None,
    model_path: Optional[str] = None,
    distance_metric: Optional[Any] = None,
    skip_vad: bool = False,
    reconcile: bool = True,
    export_praat: bool = True,
    export_manifest: bool = True,
    textgrid_filename: str = "greedy_alignment.TextGrid",
    manifest_filename: str = "alignment_manifest.json",
    emissions_extractor: Optional[ASREmissionsExtractor] = None,
) -> AlignmentOutput:
    """
    Aligns Cherokee audio against a syllabary transcript using the established
    Greedy ASR Inference + Sliding-Window DTW / Needleman-Wunsch pipeline with
    Syllabary Reconciliation.

    Args:
        audio: Audio file path, AudioSegment, or numpy array.
        transcript: Syllabary transcript (raw text string, path to .txt/.json,
                    or list of chunk strings/dicts). Handles code-switched English words.
        output_dir: Optional output directory to export Praat TextGrid and JSON manifest.
        model: Optional preloaded CherokeeASRModel instance.
        model_path: Optional model repository or checkpoint path to load if model is None.
        distance_metric: Optional custom distance metric for word alignment.
        skip_vad: Whether to skip VAD segmentation in audio emission extraction.
        reconcile: Whether to run phonetic syllabary reconciliation.
        export_praat: Whether to export a Praat .TextGrid file when output_dir is provided.
        export_manifest: Whether to export alignment_manifest.json when output_dir is provided.
        textgrid_filename: Name of the generated TextGrid file (default: greedy_alignment.TextGrid).
        manifest_filename: Name of the generated JSON manifest (default: alignment_manifest.json).
        emissions_extractor: Optional custom ASREmissionsExtractor.

    Returns:
        AlignmentOutput object with aligned chunks, word intervals, metrics, and tiers.
    """
    chunks, source_lookup = load_syllabary_transcript(transcript)

    if emissions_extractor is not None:
        extractor = emissions_extractor
    else:
        asr_model = model
        if asr_model is None:
            token = os.environ.get("HF_TOKEN", None)
            asr_model = CherokeeASRModel.from_pretrained_or_best(
                path_or_repo=model_path,
                token=token,
            )
        extractor = CherokeeASRExtractor(model=asr_model, skip_vad=skip_vad)

    audio_source_id = (
        str(audio) if isinstance(audio, (str, Path)) else "in_memory_audio"
    )
    emissions = extractor.extract(audio)

    word_aligner = NeedlemanWunschWordAligner(distance_metric=distance_metric)
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    alignment = aligner.align(
        emissions=emissions, chunks=chunks, source_id=audio_source_id
    )

    syllabary_lookup = {
        cid: meta.get("syllabary", meta.get("cherokee", meta.get("text", "")))
        for cid, meta in source_lookup.items()
    }
    syllabary_words_tier = _build_syllabary_word_tier(alignment, syllabary_lookup)

    additional_word_tiers: Dict[str, Sequence[WordInterval]] = {
        "Syllabary Words": syllabary_words_tier,
    }

    if reconcile:
        reconciled_words = reconcile_alignment_words(alignment, syllabary_lookup)
        additional_word_tiers["Reconciled Words"] = reconciled_words

    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if export_manifest:
            export_manifest_file(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=manifest_filename,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

        if export_praat:
            export_textgrid(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=textgrid_filename,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

    return alignment


def align_syllabary_ctc(
    audio: Union[str, Path, AudioSegment, np.ndarray],
    transcript: Union[str, Path, List[str], List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[CherokeeASRModel] = None,
    model_path: Optional[str] = None,
    config: Optional[CTCAlignerConfig] = None,
    cache: bool = True,
    reconcile: bool = True,
    export_praat: bool = True,
    export_manifest: bool = True,
    textgrid_filename: str = "ctc_alignment.TextGrid",
    manifest_filename: str = "alignment_manifest.json",
) -> AlignmentOutput:
    """
    Aligns Cherokee audio against a syllabary transcript using the syncope-
    and intrusion-aware CTC Segmentation engine.

    Args:
        audio: Audio file path, AudioSegment, or numpy array.
        transcript: Syllabary transcript (raw text string, path to .txt/.json,
                    or list of chunk strings/dicts). Handles code-switched English words.
        output_dir: Optional output directory to export Praat TextGrid and JSON manifest.
        model: Optional preloaded CherokeeASRModel instance.
        model_path: Optional model repository or checkpoint path to load if model is None.
        config: Optional CTCAlignerConfig instance.
        cache: Whether to cache acoustic log-probability matrices on disk.
        reconcile: Whether to run phonetic syllabary reconciliation on CTC emitted paths.
        export_praat: Whether to export a Praat .TextGrid file when output_dir is provided.
        export_manifest: Whether to export alignment_manifest.json when output_dir is provided.
        textgrid_filename: Name of the generated TextGrid file (default: ctc_alignment.TextGrid).
        manifest_filename: Name of the generated JSON manifest (default: alignment_manifest.json).

    Returns:
        AlignmentOutput object with CTC-segmented aligned chunks, word intervals, and tiers.
    """
    chunks, source_lookup = load_syllabary_transcript(transcript)

    asr_model = model
    if asr_model is None:
        token = os.environ.get("HF_TOKEN", None)
        asr_model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=model_path,
            token=token,
        )

    aligner = CTCSegmentationAligner(model=asr_model, config=config)
    audio_source_id = (
        str(audio) if isinstance(audio, (str, Path)) else "in_memory_audio"
    )

    alignment = aligner.align(
        audio_input=audio,
        chunks=chunks,
        source_id=audio_source_id,
        cache=cache,
    )

    syllabary_lookup = {
        cid: meta.get("syllabary", meta.get("cherokee", meta.get("text", "")))
        for cid, meta in source_lookup.items()
    }
    syllabary_words_tier = _build_syllabary_word_tier(alignment, syllabary_lookup)

    additional_word_tiers: Dict[str, Sequence[WordInterval]] = {
        "Syllabary Words": syllabary_words_tier,
    }

    if reconcile:
        reconciled_words = reconcile_alignment_words(alignment, syllabary_lookup)
        additional_word_tiers["Reconciled Words"] = reconciled_words

    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if export_manifest:
            export_manifest_file(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=manifest_filename,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

        if export_praat:
            export_textgrid(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=textgrid_filename,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

    return alignment


__all__ = [
    "align_syllabary_greedy",
    "align_syllabary_ctc",
]
