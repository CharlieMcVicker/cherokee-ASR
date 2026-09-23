# -*- coding: utf-8 -*-
"""
pipeline.py

End-to-end Scripture Chapter and Verse Slicing Pipeline.
Composes Tier 1 CTCSegmentationAligner (transcription.core.alignment.ctc) and Tier 2
prepare_cherokee_text (transcription.cherokee.phonotactics) with ModelOutput universal currency,
AudioChunk/AlignedChunk domain models, verse boundary slicing to 16kHz mono WAV clips,
and optional Praat TextGrid and JSON manifest exporting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydub import AudioSegment

from transcription.cherokee.models import CherokeeASRModel
from transcription.cherokee.phonotactics import prepare_cherokee_text
from transcription.alignment.ctc_aligner import (
    DEFAULT_CACHE_DIR,
    CTCSegmentationAligner,
)
from transcription.core.alignment.ctc import (
    CTCSegmentationAligner as CoreCTCSegmentationAligner,
)
from transcription.core.alignment.models import (
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    WordInterval,
)
from transcription.core.audio.segment import AudioChunk
from transcription.core.exporters.manifest import (
    export_debug_json,
    export_manifest as export_manifest_file,
)
from transcription.core.exporters.textgrid import export_textgrid
from transcription.core.models.output import ModelOutput
from transcription.pipelines.scripture.ingestion import (
    default_scripture_phonetic_normalizer,
    load_bible_chunks,
    load_chapter_transcript,
)
from transcription.syllabary_enrichment import (
    align_character_syllable,
    reconcile_phonetics,
)

logger = logging.getLogger(__name__)


def reconcile_syllabary_asr(
    syllabary_text: str,
    asr_hypothesis: str,
    toggle_tla_lha: bool = True,
    allow_liquid_lh: bool = True,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Perform syllabary/ASR reconciliation by applying phonological enrichment rules.
    """
    alignment = align_character_syllable(syllabary_text, asr_hypothesis)
    base_trans = "".join([pair[0] for pair in alignment])
    enriched_text = reconcile_phonetics(
        syllabary_text=syllabary_text,
        base_transliteration=base_trans,
        emitted_text=asr_hypothesis,
        aligned_pairs=alignment,
    )
    return enriched_text, alignment


class ScripturePipeline:
    """
    Scripture chapter audio and verse alignment & slicing pipeline.

    Orchestrates:
    1. Chapter transcript ingestion (JSON/TSV).
    2. Continuous audio feature extraction via ASR model (ModelOutput / cached lpz).
    3. Syncope- and intrusion-aware CTC segmentation composing Tier 1 CTCSegmentationAligner
       and Tier 2 Cherokee phonotactics text preparer.
    4. Slicing clean 16kHz mono WAV clips using natural verse boundary partition points
       modeled as AudioChunks.
    5. Multi-tier Praat TextGrid and JSON manifest exporting.
    """

    def __init__(
        self,
        asr_model: Optional[CherokeeASRModel] = None,
        aligner_config: Optional[CTCAlignerConfig] = None,
        ctc_aligner: Optional[Any] = None,
    ):
        self.asr_model = asr_model
        self.config = aligner_config or CTCAlignerConfig()
        if ctc_aligner is not None:
            self.ctc_aligner = ctc_aligner
        else:
            # Wire Tier 1 CTCSegmentationAligner with Tier 2 Cherokee phonotactics preparer
            self.ctc_aligner = CoreCTCSegmentationAligner(
                config=self.config,
                text_preparer=prepare_cherokee_text,
            )

    def align(
        self,
        audio: Union[str, Path, AudioSegment, np.ndarray, ModelOutput],
        transcript: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
        source_id: str = "",
        normalizer: Callable[[str], str] = default_scripture_phonetic_normalizer,
    ) -> Tuple[AlignmentOutput, Dict[str, Dict[str, Any]]]:
        """
        Align a continuous scripture chapter audio recording against its transcript.

        Args:
            audio: Chapter audio file path, AudioSegment, waveform array, or precomputed ModelOutput.
            transcript: Path to chapter JSON/TSV or loaded metadata dictionary/list.
            source_id: Optional identifier for the chapter recording.
            normalizer: Function to normalize raw phonetic text.

        Returns:
            Tuple of (AlignmentOutput, source_lookup_dict).
        """
        chunks, source_lookup = load_bible_chunks(transcript, normalizer=normalizer)
        resolved_source_id = source_id or (
            str(audio) if isinstance(audio, (str, Path)) else "chapter"
        )

        model_out: Optional[ModelOutput] = (
            audio if isinstance(audio, ModelOutput) else None
        )
        if model_out is None and self.asr_model is not None:
            if isinstance(audio, AudioSegment):
                audio_input_arr = (
                    np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                )
                model_out = self.asr_model.infer(audio_input_arr)
            elif not isinstance(audio, ModelOutput):
                model_out = self.asr_model.infer(audio)

        if model_out is not None:
            alignment = getattr(self.ctc_aligner, "align")(
                output=model_out,
                chunks=chunks,
                source_id=resolved_source_id,
                source_metadata=source_lookup,
            )
        else:
            # Check if aligner has its own extraction logic or requires ModelOutput
            if hasattr(self.ctc_aligner, "extract_logits") or hasattr(
                self.ctc_aligner, "get_logits_cached"
            ):
                # Align using aligner's audio-handling align method
                align_fn = getattr(self.ctc_aligner, "align")
                try:
                    alignment = align_fn(
                        audio_input=audio,
                        chunks=chunks,
                        source_id=resolved_source_id,
                        source_metadata=source_lookup,
                    )
                except TypeError:
                    # Generic CTCSegmentationAligner expects ModelOutput or np.ndarray
                    raise ValueError(
                        "ASR model or precomputed ModelOutput must be provided to align scripture audio."
                    )
            else:
                raise ValueError(
                    "ASR model must be provided to ScripturePipeline to align audio."
                )

        return alignment, source_lookup

    def slice_verse_audio(
        self,
        audio: Union[str, Path, AudioSegment],
        alignment: AlignmentOutput,
        output_dir: Optional[Union[str, Path]] = None,
        file_prefix: str = "",
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        skip_anomalies: bool = False,
    ) -> List[AudioChunk]:
        """
        Partition chapter audio into verse clips at natural inter-verse boundary points.
        Outputs 16kHz mono AudioChunk domain models and optionally exports WAV clips to disk.

        Args:
            audio: Audio path or AudioSegment.
            alignment: AlignmentOutput containing aligned chunks with start_sec/end_sec.
            output_dir: Optional directory to save exported .wav clips.
            file_prefix: Optional prefix for exported audio files.
            target_sample_rate: Target audio sampling rate (default: 16000).
            target_channels: Target channels count (default: 1 for mono).
            skip_anomalies: If True, skip verses that have flagged anomaly words.

        Returns:
            List of AudioChunk instances.
        """
        if isinstance(audio, (str, Path)):
            audio_seg: AudioSegment = (
                AudioSegment.from_file(str(audio))
                .set_frame_rate(target_sample_rate)
                .set_channels(target_channels)
            )
        elif isinstance(audio, AudioSegment):
            audio_seg = audio.set_frame_rate(target_sample_rate).set_channels(
                target_channels
            )
        else:
            raise TypeError(f"Unsupported audio type for slicing: {type(audio)}")

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        audio_chunks: List[AudioChunk] = []

        for idx, chunk in enumerate(alignment.aligned_chunks):
            if skip_anomalies and chunk.has_anomalies:
                continue

            start_ms = max(0, int(round(chunk.start_sec * 1000)))
            end_ms = min(
                len(audio_seg), max(start_ms, int(round(chunk.end_sec * 1000)))
            )

            # Slice verse
            verse_audio: AudioSegment = audio_seg[start_ms:end_ms]  # type: ignore

            chunk_item = AudioChunk(
                chunk_index=idx,
                audio=verse_audio,
                start_sec=round(start_ms / 1000.0, 3),
                end_sec=round(end_ms / 1000.0, 3),
            )
            audio_chunks.append(chunk_item)

            if output_dir is not None and len(verse_audio) > 0:
                name_prefix = f"{file_prefix}_" if file_prefix else ""
                clean_chunk_id = chunk.chunk_id.replace("/", "_")
                filename = f"{name_prefix}{clean_chunk_id}.wav"
                out_path = Path(output_dir) / filename
                verse_audio.export(str(out_path), format="wav")

        return audio_chunks

    def run(
        self,
        audio_path: Union[str, Path],
        transcript_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        slice_audio: bool = True,
        export_praat: bool = True,
        export_manifest: bool = True,
        debug_export: bool = False,
        reconcile: bool = True,
        skip_anomalies_for_audio: bool = False,
    ) -> Tuple[AlignmentOutput, List[AudioChunk]]:
        """
        Execute full end-to-end chapter alignment, verse slicing, and artifact export.

        Args:
            audio_path: Path to continuous chapter audio file (.wav/.mp3).
            transcript_path: Path to chapter transcript JSON/TSV file.
            output_dir: Optional directory to save Praat TextGrid, manifest, and audio slices.
            slice_audio: Whether to slice audio into 16kHz WAV verse clips.
            export_praat: Whether to export Praat TextGrid.
            export_manifest: Whether to export alignment_manifest.json.
            debug_export: Whether to export alignment_debug.json.
            reconcile: Whether to perform phonological syllabary/ASR reconciliation.
            skip_anomalies_for_audio: Skip audio export for anomaly-flagged verses.

        Returns:
            Tuple of (AlignmentOutput, List[AudioChunk]).
        """
        alignment, source_lookup = self.align(
            audio=audio_path,
            transcript=transcript_path,
            source_id=str(audio_path),
        )

        audio_chunks: List[AudioChunk] = []
        if slice_audio:
            slice_out_dir = Path(output_dir) / "split_audio" if output_dir else None
            audio_chunks = self.slice_verse_audio(
                audio=audio_path,
                alignment=alignment,
                output_dir=slice_out_dir,
                skip_anomalies=skip_anomalies_for_audio,
            )

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            additional_word_tiers = None
            if reconcile:
                reconciled_words = [
                    WordInterval(
                        word=w.emitted_word or w.word,
                        start_sec=w.start_sec,
                        end_sec=w.end_sec,
                        confidence=w.confidence,
                        flagged=w.flagged,
                        emitted_word=w.emitted_word,
                        min_char_confidence=w.min_char_confidence,
                    )
                    for c in alignment.aligned_chunks
                    for w in c.words
                ]
                additional_word_tiers = {"Reconciled Transcriptions": reconciled_words}

            if export_manifest:
                export_manifest_file(
                    alignment=alignment,
                    output_dir=out_p,
                    source_metadata=source_lookup,
                    additional_word_tiers=additional_word_tiers,
                )

            if export_praat:
                export_textgrid(
                    alignment=alignment,
                    output_dir=out_p,
                    source_metadata=source_lookup,
                    additional_word_tiers=additional_word_tiers,
                )

            if debug_export:
                export_debug_json(
                    alignment=alignment,
                    output_dir=out_p,
                )

        return alignment, audio_chunks


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
    ctc_aligner: Optional[CTCSegmentationAligner] = None,
    asr_model: Optional[CherokeeASRModel] = None,
    cache: bool = True,
    aligner_config: Optional[CTCAlignerConfig] = None,
) -> AlignmentOutput:
    """
    Procedural backward-compatible wrapper for scripture chapter alignment.

    Supports continuous chapter alignment using CTCSegmentationAligner (default, engine="ctc"),
    or DTW emissions alignment via run_alignment_pipeline (engine="dtw" or when DTW ModelOutput is provided).
    """
    use_ctc = engine.lower() == "ctc" and (
        ctc_aligner is not None
        or (distance_metric is None and emissions_extractor is None)
    )

    if use_ctc:
        chunks, source_lookup = load_bible_chunks(
            transcript_path, normalizer=default_scripture_phonetic_normalizer
        )

        aligner = ctc_aligner
        if aligner is None:
            model = asr_model
            if model is None:
                rev = model_revision or "5464d15"
                repo = model_path or "charliemcvicker/asr-cherokee"
                token = os.environ.get("HF_TOKEN", None)
                model = CherokeeASRModel.from_pretrained_or_best(
                    path_or_repo=repo,
                    revision=rev,
                    token=token,
                )
            if aligner_config is None:
                c_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
                aligner_config = CTCAlignerConfig(
                    cache=cache,
                    cache_dir=c_dir,
                )
            aligner = CTCSegmentationAligner(
                model=model,
                config=aligner_config,
            )

        # Check if aligner supports audio_input or ModelOutput
        if hasattr(aligner, "extract_logits") or hasattr(aligner, "get_logits_cached"):
            alignment = getattr(aligner, "align")(
                audio_input=audio_path,
                chunks=chunks,
                source_id=str(audio_path),
            )
        else:
            model = asr_model or getattr(aligner, "model", None)
            if model is None:
                rev = model_revision or "5464d15"
                repo = model_path or "charliemcvicker/asr-cherokee"
                token = os.environ.get("HF_TOKEN", None)
                model = CherokeeASRModel.from_pretrained_or_best(
                    path_or_repo=repo,
                    revision=rev,
                    token=token,
                )
            model_out = model.infer(audio_path)
            alignment = getattr(aligner, "align")(
                output=model_out,
                chunks=chunks,
                source_id=str(audio_path),
                source_metadata=source_lookup,
            )

        additional_word_tiers = None
        if reconcile:
            reconciled_words = [
                WordInterval(
                    word=w.emitted_word or w.word,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=w.emitted_word,
                    min_char_confidence=w.min_char_confidence,
                )
                for c in alignment.aligned_chunks
                for w in c.words
            ]
            additional_word_tiers = {"Reconciled Transcriptions": reconciled_words}

        os.makedirs(str(output_dir), exist_ok=True)

        if export_manifest:
            export_manifest_file(
                alignment=alignment,
                output_dir=output_dir,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

        if export_praat:
            export_textgrid(
                alignment=alignment,
                output_dir=output_dir,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

        if debug_export:
            export_debug_json(
                alignment=alignment,
                output_dir=output_dir,
            )

        return alignment

    # Fallback to legacy DTW alignment pipeline if custom metric/model_output provided
    from transcription.alignment.cli import run_alignment_pipeline
    from transcription.cherokee.distance import (
        PhonologicalConfusionCostMetric,
    )
    from transcription.alignment.distance_metrics import ConfusionMatrixCostMetric

    if distance_metric is None:
        cost_matrix_path = Path("runs/evaluation/confusion_cost_matrix_prebible.json")
        if not cost_matrix_path.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = (
                repo_root / "runs/evaluation/confusion_cost_matrix_prebible.json"
            )
            if candidate.exists():
                cost_matrix_path = candidate
        if cost_matrix_path.exists():
            base_metric = ConfusionMatrixCostMetric.from_json(cost_matrix_path)
            distance_metric = PhonologicalConfusionCostMetric(base_metric=base_metric)

    model_out = (
        emissions_extractor if isinstance(emissions_extractor, ModelOutput) else None
    )
    if model_out is None:
        rev = model_revision or "5464d15"
        repo = model_path or "charliemcvicker/asr-cherokee"
        token = os.environ.get("HF_TOKEN", None)
        model = asr_model or CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=repo,
            revision=rev,
            token=token,
        )
        c_dir = (
            Path(cache_dir) if cache_dir is not None else Path("runs/cache/emissions")
        )
        model_out = model.infer(audio_path, cache_dir=c_dir if cache else None)

    return run_alignment_pipeline(
        audio_path=str(audio_path),
        output_dir=str(output_dir),
        bible_metadata_path=str(transcript_path),
        export_praat=export_praat,
        export_manifest=export_manifest,
        model_path=model_path,
        skip_vad=skip_vad,
        debug_export=debug_export,
        reconcile=reconcile,
        distance_metric=distance_metric,
        model_output=model_out,
    )


__all__ = [
    "ScripturePipeline",
    "align_chapter",
    "reconcile_syllabary_asr",
]
