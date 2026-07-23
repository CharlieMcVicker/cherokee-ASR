# -*- coding: utf-8 -*-
"""
align_cli.py

Main CLI runner orchestrating audio segmentation, ground truth ingest,
ASR emissions extraction, trigram DTW alignment, and export.
"""

import argparse
import os
import sys
import numpy as np

from transcription.timestamping.audio_segmenter import segment_long_audio
from transcription.timestamping.prepare_ground_truth import (
    parse_bible_metadata,
    parse_chunk_list,
)
from transcription.timestamping.aligner import align_tokens_to_verses
from transcription.timestamping.exporter import (
    export_praat_textgrid,
    export_alignment_manifest,
)


def run_alignment_pipeline(
    audio_path: str,
    output_dir: str,
    bible_metadata_path: str = None,
    chunk_list_path: str = None,
    export_praat: bool = True,
    model_path: str = None,
) -> None:
    """Executes full alignment pipeline end-to-end."""
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

    print(f"[2/4] Segmenting audio file '{audio_path}' with VAD...")
    chunks = segment_long_audio(audio_path)
    print(f"      Generated {len(chunks)} speech chunks.")

    print(f"[3/4] Running ASR emission extraction & DTW alignment...")
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transcription.utils.model_utils import get_best_model_config
    from transcription.inference.infer import (
        calculate_word_confidences,
        greedy_inference,
    )

    token = os.environ.get("HF_TOKEN", None)
    if model_path is None:
        model_config = get_best_model_config()
        model_path = model_config.get("repo", "facebook/wav2vec2-base-960h")

    print(f"      Loading model from '{model_path}'...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path, token=token)
        model = Wav2Vec2ForCTC.from_pretrained(model_path, token=token).to(device)
    except Exception as e:
        fallback_repo = "facebook/wav2vec2-base-960h"
        print(
            f"      [Warning] Could not load '{model_path}' ({e}). Falling back to public model '{fallback_repo}'..."
        )
        model_path = fallback_repo
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)

    model.eval()

    all_tokens = []
    for c in chunks:
        # Extract audio waveform array from AudioChunk
        samples = np.array(c.audio.get_array_of_samples(), dtype=np.float32)
        if c.audio.channels > 1:
            samples = samples.reshape((-1, c.audio.channels)).mean(axis=1)
        # Normalize audio amplitude float32
        max_val = float(1 << (8 * c.audio.sample_width - 1))
        samples = samples / max_val

        input_values = processor(
            samples, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits[0]

        pred_ids = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        chunk_words = calculate_word_confidences(probs, pred_ids, processor)

        for w_info in chunk_words:
            all_tokens.append(
                {
                    "word": w_info["word"],
                    "start_time": round(c.start_sec + w_info["start_time"], 3),
                    "end_time": round(c.start_sec + w_info["end_time"], 3),
                    "confidence": w_info["confidence"],
                }
            )

    print(f"      Extracted {len(all_tokens)} raw ASR word emissions across audio.")
    alignment = align_tokens_to_verses(all_tokens, verses, audio_source=audio_path)

    print(f"[4/4] Exporting alignment results to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "alignment_manifest.json")
    export_alignment_manifest(alignment, manifest_path)
    print(f"      Saved manifest: {manifest_path}")

    if export_praat:
        textgrid_path = os.path.join(output_dir, "alignment.TextGrid")
        export_praat_textgrid(alignment, textgrid_path)
        print(f"      Saved Praat TextGrid: {textgrid_path}")

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

    args = parser.parse_args()

    bible_meta = args.bible_metadata or args.metadata

    run_alignment_pipeline(
        audio_path=args.audio,
        bible_metadata_path=bible_meta,
        chunk_list_path=args.chunk_list,
        output_dir=args.output_dir,
        export_praat=args.export_praat,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
