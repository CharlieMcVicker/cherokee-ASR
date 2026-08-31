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

from transcription.alignment.adapters.inbound import (
    BibleMetadataVerseAdapter,
    GenericChunkListAdapter,
)
from transcription.alignment.adapters.outbound import (
    DebugJsonAdapter,
    ManifestJsonAdapter,
    PraatTextGridAdapter,
)
from transcription.alignment.core.sliding_window import SlidingWindowDTWAligner
from transcription.alignment.domain.models import AlignmentOutput
from transcription.alignment.pipeline import AlignmentPipeline
from transcription.alignment.strategies.extractors import CherokeeASRExtractor
from transcription.alignment.strategies.reconciliation import (
    CherokeeSyllabaryReconciliationStrategy,
)


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
    """Prepares ports, strategies, and adapters, and runs the AlignmentPipeline."""
    if bible_metadata_path:
        print(
            f"[1/4] Ingesting Bible ground-truth metadata from '{bible_metadata_path}'..."
        )
        chunk_adapter = BibleMetadataVerseAdapter.make_default(path=bible_metadata_path)
    elif chunk_list_path:
        print(f"[1/4] Ingesting ground-truth chunk list from '{chunk_list_path}'...")
        chunk_adapter = GenericChunkListAdapter.make_default(path=chunk_list_path)
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
    engine = SlidingWindowDTWAligner.make_default(
        reconciliation_strategy=(
            CherokeeSyllabaryReconciliationStrategy() if reconcile else None
        )
    )

    exporters = []
    if export_manifest:
        exporters.append((ManifestJsonAdapter(), "alignment_manifest.json"))
    if export_praat:
        exporters.append((PraatTextGridAdapter.make_default(), "alignment.TextGrid"))
    if debug_export:
        exporters.append((DebugJsonAdapter(), "alignment_debug.json"))

    pipeline = AlignmentPipeline(
        chunk_adapter=chunk_adapter,
        extractor=extractor,
        engine=engine,
        exporters=exporters,
    )

    print(f"[4/4] Executing alignment and exporting artifacts to '{output_dir}'...")
    alignment = pipeline.run(
        audio_input=audio_path,
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
