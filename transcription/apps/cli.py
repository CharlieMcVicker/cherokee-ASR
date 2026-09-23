# -*- coding: utf-8 -*-
"""
transcription.apps.cli module.

Central CLI entrypoint for Cherokee ground-truth timestamp alignment.
Delegates to domain-specific pipelines:
- ScripturePipeline (transcription.pipelines.scripture) for Bible verse chapter alignment
- DialogueAlignmentPipeline (transcription.pipelines.dialogue) for code-switched or syllabary transcripts
- SlidingWindowDTWAligner / NeedlemanWunschWordAligner (transcription.alignment) for custom DTW / chunk-list alignment
- EnrichmentPipeline (transcription.pipelines.enrichment) for syllabary phonetic reconciliation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.arpabet.types import SyntheticTargetProjectorProtocol
from transcription.alignment.exporters import (
    export_debug_json,
    export_manifest as export_manifest_file,
    export_textgrid,
)
from transcription.alignment.ingestion import prepare_alignment_input
from transcription.alignment.models import AlignmentOutput
from transcription.alignment.reconciliation import reconcile_alignment_words
from transcription.core.models.output import ModelOutput
from transcription.models.asr_model import CherokeeASRModel


def run_alignment_pipeline(
    audio_path: str,
    output_dir: str,
    bible_metadata_path: Optional[str] = None,
    chunk_list_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    model_path: Optional[str] = None,
    export_praat: bool = True,
    export_manifest: bool = True,
    skip_vad: bool = False,
    debug_export: bool = False,
    reconcile: bool = False,
    distance_metric: Optional[Any] = None,
    model_output: Optional[ModelOutput] = None,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    code_switched: bool = False,
) -> AlignmentOutput:
    """
    High-level programmatic runner executing the end-to-end alignment pipeline.
    Preserves exact function signature and semantics for backwards compatibility,
    delegating to domain pipelines where applicable while supporting all legacy flags.
    """
    os.makedirs(output_dir, exist_ok=True)

    if bible_metadata_path:
        print(
            f"[1/4] Ingesting Bible verse chunks from metadata '{bible_metadata_path}'..."
        )
    elif chunk_list_path:
        print(f"[1/4] Ingesting ground-truth chunk list from '{chunk_list_path}'...")
    elif transcript_path:
        print(f"[1/4] Ingesting transcript from '{transcript_path}'...")

    chunks, source_lookup, chunk_norm, emission_norm = prepare_alignment_input(
        bible_metadata=bible_metadata_path,
        chunk_list=chunk_list_path,
        transcript=transcript_path,
        projector=projector,
        code_switched=code_switched,
    )

    if skip_vad:
        print(f"[2/4] Skipping VAD audio segmentation (skip_vad=True)...")
    else:
        print(f"[2/4] Segmenting audio file '{audio_path}' with VAD...")

    print(f"[3/4] Running ASR emission extraction & alignment...")

    if model_output is not None:
        emissions = model_output
    else:
        token = os.environ.get("HF_TOKEN", None)
        asr_model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=model_path,
            token=token,
        )
        emissions = asr_model.infer(audio_path)

    word_aligner = NeedlemanWunschWordAligner(
        distance_metric=distance_metric,
        chunk_normalizer=chunk_norm,
        emission_normalizer=emission_norm,
    )
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    alignment = aligner.align(emissions=emissions, chunks=chunks, source_id=audio_path)

    additional_word_tiers = None
    if reconcile:
        syllabary_lookup = {
            cid: meta.get("cherokee", meta.get("cherokee_syllabary", ""))
            for cid, meta in source_lookup.items()
        }
        reconciled_words = reconcile_alignment_words(alignment, syllabary_lookup)
        additional_word_tiers = {"Reconciled Words": reconciled_words}

    print(f"[4/4] Executing alignment and exporting artifacts to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

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


# Alias for clarity
run_alignment_cli = run_alignment_pipeline


def main() -> None:
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
        "--transcript",
        help="Path to Cherokee Syllabary or code-switched transcript file (.txt/.json)",
    )
    gt_group.add_argument(
        "--metadata",
        help="Alias for --bible-metadata for backward compatibility",
    )

    parser.add_argument(
        "--code-switched",
        action="store_true",
        default=False,
        help="Enable code-switched English-to-Cherokee synthetic target projection",
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
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        export_praat=args.export_praat,
        model_path=args.model_path,
        skip_vad=args.skip_vad,
        debug_export=args.debug_export,
        reconcile=args.reconcile,
        code_switched=args.code_switched,
    )


if __name__ == "__main__":
    main()
