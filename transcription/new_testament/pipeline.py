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
from transcription.alignment.distance_metrics import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CachedASREmissionsExtractor,
    CherokeeASRExtractor,
)
from transcription.alignment.models import AlignmentOutput
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
    model_path: Optional[str] = None,
    skip_vad: bool = False,
    reconcile: bool = True,
    distance_metric: Optional[DistanceMetric] = None,
    emissions_extractor: Optional[ASREmissionsExtractor] = None,
    model_revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> AlignmentOutput:
    """
    Align a New Testament audio recording with its syllabary transcript end-to-end.

    Delegates directly to the core timestamping alignment pipeline (run_alignment_pipeline),
    performing VAD, ground-truth ingest, ASR CTC emissions extraction, DTW alignment,
    reconciliation, and automatic Praat TextGrid / alignment manifest export.
    """
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
        asr_model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=repo,
            revision=rev,
            token=token,
        )
        base_extractor = CherokeeASRExtractor(model=asr_model, skip_vad=skip_vad)
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
        model_path=model_path,
        skip_vad=skip_vad,
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
