"""
New Testament Pipeline Module for Audio-Transcript Alignment and Syllabary/ASR Reconciliation.

Imports and integrates VAD segmentation, timestamp/character alignment, and syllabary enrichment.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transcription.alignment.cli import run_alignment_pipeline
from transcription.alignment.calibrated_distance_metrics import (
    PhonologicalConfusionCostMetric,
)
from transcription.alignment.ctc_aligner import (
    CTCSegmentationAligner,
    DEFAULT_CACHE_DIR,
)
from transcription.alignment.distance_metrics import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
)
from transcription.alignment.exporters import (
    export_debug_json,
    export_manifest as export_manifest_file,
    export_textgrid,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CachedASREmissionsExtractor,
    CherokeeASRExtractor,
)
from transcription.alignment.ingestion import load_bible_chunks
from transcription.alignment.models import (
    AlignmentOutput,
    CTCAlignerConfig,
    WordInterval,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.alignment.reconciliation import reconcile_alignment_words
from transcription.models.asr_model import CherokeeASRModel
from transcription.syllabary_enrichment import (
    align_character_syllable,
    reconcile_phonetics,
)


def load_chapter_transcript(
    transcript_path: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    """
    Load a single New Testament chapter transcript JSON file.

    Returns a mapping of verse_id -> verse data dict containing keys like
    'cherokee', 'phonetic', 'english', etc.
    """
    path = Path(transcript_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, str]] = json.load(f)
    return data


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
    distance_metric: Optional[DistanceMetric] = None,
    emissions_extractor: Optional[ASREmissionsExtractor] = None,
    model_revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    engine: str = "ctc",
    ctc_aligner: Optional[CTCSegmentationAligner] = None,
    asr_model: Optional[CherokeeASRModel] = None,
    cache: bool = True,
    aligner_config: Optional[CTCAlignerConfig] = None,
) -> AlignmentOutput:
    """
    Align a New Testament audio recording with its syllabary transcript end-to-end.

    Supports continuous chapter alignment using CTCSegmentationAligner (default, engine="ctc"),
    or DTW emissions alignment via run_alignment_pipeline (engine="dtw" or when DTW extractors are provided).

    Args:
        audio_path: Path to continuous chapter audio file (.wav/.mp3).
        transcript_path: Path to chapter transcript JSON file.
        output_dir: Directory to save exported Praat TextGrid and alignment JSON.
        export_praat: Whether to export Praat TextGrids.
        export_manifest: Whether to export alignment_manifest.json.
        debug_export: Whether to export alignment_debug.json.
        model_path: HuggingFace model repo or local checkpoint path.
        skip_vad: Bypass VAD segmentation (for legacy DTW engine).
        reconcile: Whether to perform syllabary/ASR phonological reconciliation.
        distance_metric: Custom distance metric (for legacy DTW engine).
        emissions_extractor: Custom emissions extractor (for legacy DTW engine).
        model_revision: HuggingFace model revision tag or commit hash.
        cache_dir: Directory to store/load cached emissions or CTC logits.
        engine: Alignment engine ("ctc" or "dtw"). Defaults to "ctc".
        ctc_aligner: Optional pre-instantiated CTCSegmentationAligner.
        asr_model: Optional pre-instantiated CherokeeASRModel.
        cache: Whether to use disk caching for CTC logits.
        aligner_config: Optional strongly-typed CTCAlignerConfig for CTC segmentation.

    Returns:
        AlignmentOutput object containing aligned chunks, words, and metrics.
    """
    use_ctc = engine.lower() == "ctc" and (
        ctc_aligner is not None
        or (distance_metric is None and emissions_extractor is None)
    )

    if use_ctc:
        chunks, source_lookup = load_bible_chunks(
            transcript_path, normalizer=normalize_phonetics_for_alignment
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

        alignment = aligner.align(
            audio_input=audio_path,
            chunks=chunks,
            source_id=str(audio_path),
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

    # Legacy DTW alignment fallback
    if distance_metric is None:
        cost_matrix_path = Path("runs/evaluation/confusion_cost_matrix_prebible.json")
        if not cost_matrix_path.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            candidate = (
                repo_root / "runs/evaluation/confusion_cost_matrix_prebible.json"
            )
            if candidate.exists():
                cost_matrix_path = candidate
        if cost_matrix_path.exists():
            base_metric = ConfusionMatrixCostMetric.from_json(cost_matrix_path)
            distance_metric = PhonologicalConfusionCostMetric(base_metric=base_metric)

    if emissions_extractor is None:
        rev = model_revision or "5464d15"
        repo = model_path or "charliemcvicker/asr-cherokee"
        token = os.environ.get("HF_TOKEN", None)
        model = asr_model or CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=repo,
            revision=rev,
            token=token,
        )
        base_extractor = CherokeeASRExtractor(model=model, skip_vad=skip_vad)
        c_dir = (
            Path(cache_dir) if cache_dir is not None else Path("runs/cache/emissions")
        )
        c_prefix = f"charliemcvicker_asr-cherokee_{rev}"
        emissions_extractor = CachedASREmissionsExtractor(
            extractor=base_extractor,
            cache_dir=c_dir,
            cache_key_prefix=c_prefix,
        )

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
        emissions_extractor=emissions_extractor,
    )


def reconcile_syllabary_asr(
    syllabary_text: str,
    asr_hypothesis: str,
    toggle_tla_lha: bool = True,
    allow_liquid_lh: bool = True,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Perform syllabary/ASR reconciliation by applying phonological enrichment rules.

    Leverages enrich_syllabary from transcription.syllabary_enrichment.

    Args:
        syllabary_text: Ground truth Cherokee syllabary text.
        asr_hypothesis: Raw ASR model hypothesis text.
        toggle_tla_lha: Phonological toggle for tla/tli -> lha/lhi.
        allow_liquid_lh: Enable voiceless/aspirated lateral lh enrichment.

    Returns:
        Tuple of (enriched_transcription, list_of_transformation_notes).
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
