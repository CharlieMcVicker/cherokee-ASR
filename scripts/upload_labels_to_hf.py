#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/upload_labels_to_hf.py

Lightweight helper to mirror updated CSV splits and vocabulary JSON files
to Hugging Face Hub without touching or re-uploading the 462 MB audio tarball.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("upload_labels_to_hf")

DEFAULT_REPO_ID = "charliemcvicker/cherokee-asr-data"
DEFAULT_PROCESSED_DIR = Path("data/training/processed")
LABEL_FILENAMES: Sequence[str] = (
    "chr-wav2vec2-train.csv",
    "chr-wav2vec2-valid.csv",
    "chr-wav2vec2-test.csv",
    "bible-wav2vec2-train.csv",
    "bible-wav2vec2-valid.csv",
    "bible-wav2vec2-test.csv",
    "split_audio_syl_target.csv",
    "vocab.json",
    "vocab_tone.json",
)


def upload_labels(
    repo_id: str = DEFAULT_REPO_ID,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    token: str | None = None,
    revision: str | None = None,
) -> list[str]:
    from huggingface_hub import HfApi

    hf_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    api = HfApi(token=hf_token)
    uploaded: list[str] = []

    for fname in LABEL_FILENAMES:
        local_path = processed_dir / fname
        if not local_path.exists():
            logger.warning("Skipping missing file: %s", local_path)
            continue
        path_in_repo = f"training/processed/{fname}"
        logger.info("Uploading %s -> %s:%s", local_path, repo_id, path_in_repo)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=f"Update dual-head label/vocab artifact: {fname}",
        )
        uploaded.append(fname)

    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload updated CSV label splits and vocab JSONs to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Local processed directory (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face user access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Target branch or revision on Hugging Face Hub",
    )

    args = parser.parse_args()
    upload_labels(
        repo_id=args.repo_id,
        processed_dir=args.processed_dir,
        token=args.token,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
