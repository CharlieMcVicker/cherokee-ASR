#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_julie.py

Runs speech-to-text inference on a single audio file using the fine-tuned Wav2Vec2 model.
"""

import os

# Enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from transcription.utils.model_utils import get_best_model_config
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

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if os.path.exists(args.processor):
        print(f"Loading processor from local path: {args.processor}...")
    else:
        print(
            f"Loading processor from Hugging Face Hub: {args.processor} (revision: {args.revision})..."
        )
    processor = Wav2Vec2Processor.from_pretrained(
        args.processor, token=token, revision=args.revision
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading model from local path: {args.checkpoint}...")
    else:
        print(
            f"Loading model from Hugging Face Hub: {args.checkpoint} (revision: {args.revision})..."
        )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint, token=token, revision=args.revision
    )
    model.eval()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    model.to(device)

    print(f"Transcribing audio file: {args.audio_path}...")
    result = infer_single_audio(model, processor, args.audio_path, device=device)

    print("\n" + "=" * 60)
    print("GREEDY DECODING PREDICTIONS:")
    print(f"  Transcription: {result['text']}")
    print(f"  Confidence:    {result['confidence']:.4f} ({result['confidence']:.2%})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
