# -*- coding: utf-8 -*-
"""
cli.py

Lightweight CLI entrypoint for the modular Cherokee alignment pipeline.
"""

import argparse
import os
import sys
from typing import Any, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.exporters import (
    export_debug_json as write_debug_json,
    export_manifest as write_manifest_json,
    export_textgrid as write_textgrid_file,
)
from transcription.alignment.extractors import CherokeeASRExtractor
from transcription.alignment.ingestion import load_bible_chunks, load_generic_chunks
from transcription.alignment.models import AlignmentOutput
from transcription.alignment.normalizers import normalize_text_for_alignment
from transcription.alignment.reconciliation import reconcile_alignment


def run_alignment_pipeline(
    audio_path: str,
    output_dir: str,
    bible_metadata_path: Optional[str] = None,
    chunk_list_path: Optional[str] = None,
    export_praat: bool = True,
    export_manifest: bool = True,
    model_path: Optional[str] = None,
    skip_vad: bool = False,
    debug_export: bool = False,
    reconcile: bool = False,
) -> AlignmentOutput:
    """Runs the streamlined Cherokee alignment pipeline and exports artifacts."""
    if bible_metadata_path:
        print(
            f"[1/4] Ingesting Bible ground-truth metadata from '{bible_metadata_path}'..."
        )
        chunks, source_lookup = load_bible_chunks(bible_metadata_path)
    elif chunk_list_path:
        print(f"[1/4] Ingesting ground-truth chunk list from '{chunk_list_path}'...")
        chunks, source_lookup = load_generic_chunks(chunk_list_path)
    else:
        raise ValueError("Either --bible-metadata or --chunk-list must be provided.")

    if skip_vad:
        print(f"[2/4] Skipping VAD audio segmentation (skip_vad=True)...")
    else:
        print(f"[2/4] Segmenting audio file '{audio_path}' with VAD...")

    print(f"[3/4] Running ASR emission extraction & alignment...")
    from transcription.models.asr_model import CherokeeASRModel
    from transcription.utils.model_utils import get_best_model_config

    token = os.environ.get("HF_TOKEN", None)
    model_revision: Optional[str] = None
    if model_path is None:
        model_config = get_best_model_config()
        model_path = str(model_config.get("repo", "facebook/wav2vec2-base-960h"))
        model_revision = model_config.get("revision", None)

    try:
        asr_model = CherokeeASRModel.from_pretrained(
            path_or_repo=model_path,
            revision=model_revision,
            token=token,
        )
    except Exception as e:
        fallback_repo = "facebook/wav2vec2-base-960h"
        print(
            f"      [Warning] Could not load '{model_path}' ({e}). Falling back to public model '{fallback_repo}'..."
        )
        asr_model = CherokeeASRModel.from_pretrained(
            path_or_repo=fallback_repo,
            token=token,
        )

    extractor = CherokeeASRExtractor(model=asr_model, skip_vad=skip_vad)
    emissions = extractor.extract(audio_path)

    word_aligner = NeedlemanWunschWordAligner(
        chunk_normalizer=normalize_text_for_alignment,
        emission_normalizer=normalize_text_for_alignment,
    )
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    alignment = aligner.align(emissions=emissions, chunks=chunks, source_id=audio_path)

    if reconcile:
        syllabary_lookup = {
            cid: meta.get("cherokee", meta.get("cherokee_syllabary", ""))
            for cid, meta in source_lookup.items()
        }
        alignment = reconcile_alignment(alignment, syllabary_lookup)

    print(f"[4/4] Executing alignment and exporting artifacts to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    if export_manifest:
        write_manifest_json(
            alignment=alignment,
            output_dir=output_dir,
            source_metadata=source_lookup,
        )

    if export_praat:
        write_textgrid_file(
            alignment=alignment,
            output_dir=output_dir,
            source_metadata=source_lookup,
        )

    if debug_export:
        write_debug_json(
            alignment=alignment,
            output_dir=output_dir,
        )

    if alignment.metrics:
        m = alignment.metrics
        print("\n--- Alignment Metrics Summary ---")
        print(
            f"  Matched Chunks       : {m.matched_chunks} / {m.total_chunks} ({m.match_ratio*100:.1f}%)"
        )
        print(f"  Mean Distance Score  : {m.mean_distance_score:.4f}")
        print(
            f"  Matched GT vs Emitted: {m.total_ground_truth_chars} vs {m.total_emitted_chars} chars"
        )

    print("\nAlignment pipeline complete!")
    return alignment


def main():
    parser = argparse.ArgumentParser(
        description="Cherokee Ground-Truth Timestamp Alignment CLI"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to input audio file (.wav/.mp3)"
    )

    # Ground truth ingest options
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--bible-metadata",
        help="Path to Bible verse metadata JSON file (key-value dict format)",
    )
    gt_group.add_argument(
        "--chunk-list",
        help="Path to target chunk list JSON file (list of segment dicts)",
    )
    gt_group.add_argument(
        "--metadata",
        help="Alias for --bible-metadata for backward compatibility",
    )

    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--export-praat",
        action="store_true",
        default=True,
        help="Export Praat .TextGrid file",
    )
    parser.add_argument(
        "--model-path", default=None, help="Path to custom Wav2Vec2 model directory"
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        default=False,
        help="Bypass VAD audio segmentation when audio is pre-cut",
    )
    parser.add_argument(
        "--debug-export",
        action="store_true",
        default=False,
        help="Save additional debug alignment output file",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        default=False,
        help="Perform syllabary/ASR phonological reconciliation",
    )

    args = parser.parse_args()

    bible_meta = args.bible_metadata or args.metadata

    run_alignment_pipeline(
        audio_path=args.audio,
        bible_metadata_path=bible_meta,
        chunk_list_path=args.chunk_list,
        output_dir=args.output_dir,
        export_praat=args.export_praat,
        model_path=args.model_path,
        skip_vad=args.skip_vad,
        debug_export=args.debug_export,
        reconcile=args.reconcile,
    )


if __name__ == "__main__":
    main()
