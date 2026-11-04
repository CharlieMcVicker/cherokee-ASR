#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/push_checkpoint_to_hub.py

Uploads a specific training checkpoint (weights, config, and processor)
along with evaluation logs to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi
from transformers import Wav2Vec2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a specific local checkpoint and its processor to Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/workspace/output_w2v2/wav2vec2-large-xlsr/checkpoint-7600"),
        help="Path to the checkpoint folder (default: /workspace/output_w2v2/wav2vec2-large-xlsr/checkpoint-7600)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="charliemcvicker/bible-augmentation-checkpoint-7600",
        help="Hugging Face repo ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face user access token (defaults to HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=Path("/workspace/output_w2v2/wav2vec2-large-xlsr"),
        help="Directory containing processor/tokenizer files (default: /workspace/output_w2v2/wav2vec2-large-xlsr)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/workspace/output_w2v2/logs-wav2vec2-res"),
        help="Directory containing evaluation logs and rankings (default: /workspace/output_w2v2/logs-wav2vec2-res)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/workspace/output_w2v2/export_hub_checkpoint"),
        help="Temporary export staging directory (default: /workspace/output_w2v2/export_hub_checkpoint)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository instead of private",
    )
    return parser.parse_args()


def push_checkpoint(
    checkpoint_dir: Path,
    repo_id: str,
    processor_dir: Path,
    logs_dir: Path,
    export_dir: Path,
    token: str | None = None,
    private: bool = True,
) -> None:
    hf_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        print(
            "ERROR: Hugging Face token required. Provide --token or set HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not checkpoint_dir.exists():
        print(
            f"ERROR: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"=== Staging Checkpoint for Hub Upload ===")
    print(f"Source Checkpoint : {checkpoint_dir}")
    print(f"Target Repo       : {repo_id}")
    print(f"Staging Directory : {export_dir}")

    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy model weights and configuration
    ignored_patterns = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "scaler.pt",
    }
    copied_count = 0
    for item in checkpoint_dir.iterdir():
        if item.name not in ignored_patterns and item.is_file():
            shutil.copy2(item, export_dir / item.name)
            copied_count += 1
    print(f"Copied {copied_count} model files to staging directory.")

    # 2. Save processor & tokenizer configuration
    if processor_dir.exists():
        print(f"Loading processor from {processor_dir}...")
        try:
            processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
            processor.save_pretrained(str(export_dir))
            print("Processor and tokenizer configuration saved to staging directory.")
        except Exception as exc:
            print(f"Warning: Failed to load processor from {processor_dir}: {exc}")
    else:
        print(
            f"Warning: Processor dir {processor_dir} not found; proceeding without external processor."
        )

    # 3. Create repo and upload folder
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    print(f"Uploading model folder to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {checkpoint_dir.name} weights and processor",
    )
    print("Model weights successfully uploaded.")

    # 4. Upload evaluation logs if available
    if logs_dir.exists():
        print(f"Uploading evaluation logs from {logs_dir}...")
        try:
            api.upload_folder(
                folder_path=str(logs_dir),
                path_in_repo="eval_logs",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add evaluation logs and rankings for {checkpoint_dir.name}",
            )
            print("Evaluation logs successfully uploaded.")
        except Exception as exc:
            print(f"Warning: Failed to upload evaluation logs: {exc}")

    print(
        f"\nSUCCESS: Checkpoint successfully pushed to https://huggingface.co/{repo_id}"
    )


def main() -> None:
    args = parse_args()
    push_checkpoint(
        checkpoint_dir=args.checkpoint.resolve(),
        repo_id=args.repo_id,
        processor_dir=args.processor_dir.resolve(),
        logs_dir=args.logs_dir.resolve(),
        export_dir=args.export_dir.resolve(),
        token=args.token,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
