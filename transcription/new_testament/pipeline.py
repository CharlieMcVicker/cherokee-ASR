"""
New Testament Pipeline Module for Audio-Transcript Alignment and Syllabary/ASR Reconciliation.

Imports and integrates VAD segmentation, timestamp/character alignment, and syllabary enrichment.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transcription.syllabary_enrichment import reconcile_phonetics


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
) -> Any:
    """
    Align a New Testament audio recording with its syllabary transcript end-to-end.

    Delegates directly to the core timestamping alignment pipeline (run_alignment_pipeline),
    performing VAD, ground-truth ingest, ASR CTC emissions extraction, DTW alignment,
    and automatic Praat TextGrid / alignment manifest export.
    """
    from transcription.timestamping.align_cli import run_alignment_pipeline

    return run_alignment_pipeline(
        audio_path=str(audio_path),
        output_dir=str(output_dir),
        bible_metadata_path=str(transcript_path),
        export_praat=export_praat,
        model_path=model_path,
        skip_vad=skip_vad,
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
    from transcription.syllabary_enrichment import align_character_syllable

    alignment = align_character_syllable(syllabary_text, asr_hypothesis)
    base_trans = "".join([pair[0] for pair in alignment])
    enriched_text = reconcile_phonetics(
        syllabary_text=syllabary_text,
        base_transliteration=base_trans,
        emitted_text=asr_hypothesis,
        aligned_pairs=alignment,
    )
    return enriched_text, alignment
