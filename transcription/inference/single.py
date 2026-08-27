#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single.py

Runs speech-to-text inference on a single audio file using the fine-tuned Wav2Vec2 model.
Uses centralized greedy decoding.
"""

import os

# Enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import torch
from transcription.utils.model_utils import get_best_model_config, get_model
from transcription.inference.infer import infer_single_audio


def main():
    model_config = get_best_model_config()
    default_repo = model_config["repo"]
    default_revision = model_config["revision"]

    parser = argparse.ArgumentParser(
        description="Run Wav2Vec2 ASR inference on a single audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (WAV, FLAC, MP3, etc.).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_repo,
        help="Path to model checkpoint or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=default_repo,
        help="Path to saved processor or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Hub authentication token.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=default_revision,
        help="Specific Hugging Face Hub commit hash, branch, or tag.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file '{args.audio_path}' not found.")
        sys.exit(1)

    print(
        f"Loading model and processor: {args.checkpoint} (revision: {args.revision})..."
    )
    model, processor, device = get_model(
        path_or_repo=args.checkpoint,
        revision=args.revision,
        processor_path=args.processor,
        token=args.hf_token,
    )

    print(f"Using device: {device}")

    print(f"Transcribing audio file: {args.audio_path}...")
    res = infer_single_audio(model, processor, args.audio_path, device=device)
    if isinstance(res, list):
        res = res[0]

    text = str(res["text"])
    confidence = float(res["confidence"])

    print("\n" + "=" * 60)
    print("GREEDY DECODING PREDICTIONS:")
    print(f"  Transcription: {text}")
    print(f"  Confidence:    {confidence:.4f} ({confidence:.2%})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
