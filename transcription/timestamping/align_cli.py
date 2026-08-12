# -*- coding: utf-8 -*-
"""
align_cli.py

Main CLI runner orchestrating audio segmentation, ground truth ingest,
ASR emissions extraction, trigram DTW alignment, and export.
"""

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Optional, Any, Dict
import numpy as np

from transcription.timestamping.audio_segmenter import segment_long_audio
from transcription.timestamping.prepare_ground_truth import (
    parse_bible_metadata,
    parse_chunk_list,
)
from transcription.timestamping.aligner import AlignmentResult
from transcription.timestamping.exporter import (
    export_praat_textgrid,
    export_alignment_manifest,
)


def run_alignment_pipeline(
    audio_path: str,
    output_dir: str,
    bible_metadata_path: Optional[str] = None,
    chunk_list_path: Optional[str] = None,
    export_praat: bool = True,
    model_path: Optional[str] = None,
    skip_vad: bool = False,
    debug_export: bool = False,
    reconcile: bool = False,
) -> AlignmentResult:
    """Executes full alignment pipeline end-to-end using align_audio_segment."""
    if bible_metadata_path:
        print(
            f"[1/4] Ingesting Bible ground-truth metadata from '{bible_metadata_path}'..."
        )
        verses = parse_bible_metadata(bible_metadata_path)
    elif chunk_list_path:
        print(f"[1/4] Ingesting ground-truth chunk list from '{chunk_list_path}'...")
        verses = parse_chunk_list(chunk_list_path)
    else:
        raise ValueError("Either --bible-metadata or --chunk-list must be provided.")

    print(f"      Parsed {len(verses)} ground-truth segment entries.")

    if skip_vad:
        print(f"[2/4] Skipping VAD audio segmentation (skip_vad=True)...")
    else:
        print(f"[2/4] Segmenting audio file '{audio_path}' with VAD...")

    print(f"[3/4] Running ASR emission extraction & DTW alignment...")
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transcription.utils.model_utils import get_best_model_config

    token = os.environ.get("HF_TOKEN", None)
    model_revision: Optional[str] = None
    if model_path is None:
        model_config = get_best_model_config()
        model_path = str(model_config.get("repo", "facebook/wav2vec2-base-960h"))
        model_revision = model_config.get("revision", None)

    print(f"      Loading model from '{model_path}'...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    load_kwargs: Dict[str, Any] = {}
    if token:
        load_kwargs["token"] = token
    if model_revision:
        load_kwargs["revision"] = model_revision

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path, **load_kwargs)
        model_obj: Any = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs)
        model = model_obj.to(device)
    except Exception as e:
        fallback_repo = "facebook/wav2vec2-base-960h"
        print(
            f"      [Warning] Could not load '{model_path}' ({e}). Falling back to public model '{fallback_repo}'..."
        )
        model_path = fallback_repo
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        fallback_obj: Any = Wav2Vec2ForCTC.from_pretrained(model_path)
        model = fallback_obj.to(device)

    model.eval()

    from transcription.timestamping.aligner import align_audio_segment

    alignment = align_audio_segment(
        audio_input=audio_path,
        verses=verses,
        model_or_fn=model,
        processor=processor,
        audio_source=audio_path,
        skip_vad=skip_vad,
        reconcile=reconcile,
    )

    print(f"[4/4] Exporting alignment results to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "alignment_manifest.json")
    export_alignment_manifest(alignment, manifest_path)
    print(f"      Saved manifest: {manifest_path}")

    if export_praat:
        textgrid_path = os.path.join(output_dir, "alignment.TextGrid")
        export_praat_textgrid(alignment, textgrid_path)
        print(f"      Saved Praat TextGrid: {textgrid_path}")

    if debug_export:
        debug_path = os.path.join(output_dir, "alignment_debug.json")
        import json

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "audio_source": alignment.audio_source,
                    "raw_tokens": alignment.raw_tokens,
                    "verse_count": len(alignment.verses),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"      Saved debug export: {debug_path}")

    if alignment.metrics:
        m = alignment.metrics
        print("\n--- Alignment Metrics Summary ---")
        print(
            f"  Matched GT Verses    : {m.matched_verses} / {m.total_verses} ({m.matched_verse_ratio*100:.1f}%)"
        )
        print(
            f"  Matched-Verse CER    : {m.overall_cer:.4f} ({m.overall_cer*100:.2f}%)"
        )
        print(f"  Mean Verse CER       : {m.mean_verse_cer:.4f}")
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
