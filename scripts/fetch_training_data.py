#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fetch_training_data.py

Idempotent training data fetcher for remote container startup (Vast.ai, RunPod, Docker).
Downloads packaged training dataset from Hugging Face Hub using snapshot_download.
Skips download if destination is already populated unless --force is specified.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_training_data")

DEFAULT_REPO_ID = "charliemcvicker/cherokee-asr-data"
DEFAULT_DEST = Path("data")


def is_already_populated(dest: Path) -> bool:
    """
    Check if the required dataset files already exist in destination.
    """
    key_indicators = [
        dest / "training" / "processed" / "chr-wav2vec2-train.csv",
        dest / "training" / "processed" / "bible-wav2vec2-train.csv",
        dest / "training" / "processed" / "sentence_audio",
        dest / "projects" / "cherokee_new_testament" / "split_audio",
    ]
    return all(p.exists() for p in key_indicators)


def fetch_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    dest: Path = DEFAULT_DEST,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
    preserve_local_csvs: bool = False,
) -> Path:
    dest_path = dest.resolve()

    if not force and not preserve_local_csvs and is_already_populated(dest_path):
        logger.info("Dataset already populated at %s. Skipping download.", dest_path)
        return dest_path

    # Snapshot local CSV and JSON label/vocab files before extraction if preserve_local_csvs is set
    saved_local_files: dict[Path, bytes] = {}
    if preserve_local_csvs:
        processed_dir = dest_path / "training" / "processed"
        if processed_dir.exists():
            for ext in ("*.csv", "*.json"):
                for fpath in processed_dir.glob(ext):
                    if fpath.is_file():
                        saved_local_files[fpath] = fpath.read_bytes()
            logger.info(
                "Preserving %d local CSV/JSON files in %s across archive extraction.",
                len(saved_local_files),
                processed_dir,
            )

    def _restore_preserved_files() -> None:
        for fpath, content in saved_local_files.items():
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(content)
        if saved_local_files:
            logger.info(
                "Restored %d local CSV/JSON files after dataset fetch.",
                len(saved_local_files),
            )

    logger.info(
        "Fetching dataset %s from Hugging Face Hub to %s...", repo_id, dest_path
    )
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
        sys.exit(1)

    hf_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    # Strategy 1: Fast single-archive download (avoids 1,000 requests/5min HF rate limit)
    try:
        import tarfile

        logger.info(
            "Attempting fast single-archive download (cherokee_asr_data.tar.gz)..."
        )
        archive_path = hf_hub_download(
            repo_id=repo_id,
            filename="cherokee_asr_data.tar.gz",
            repo_type="dataset",
            revision=revision,
            token=hf_token,
        )
        logger.info(
            "Archive downloaded: %s. Extracting to %s...", archive_path, dest_path
        )
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_path)
        _restore_preserved_files()
        logger.info("Dataset successfully extracted to %s", dest_path)
        return dest_path
    except (EntryNotFoundError, Exception) as exc:
        logger.warning(
            "Single archive download failed or not available (%s). Falling back to snapshot_download...",
            exc,
        )

    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest_path),
        token=hf_token,
    )
    _restore_preserved_files()

    logger.info("Dataset successfully downloaded to %s", downloaded_dir)
    return Path(downloaded_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch remote training dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
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
        help="Optional branch or commit hash revision",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset directory is already populated",
    )
    parser.add_argument(
        "--preserve-local-csvs",
        action="store_true",
        help="Preserve local CSV and vocabulary JSON files in data/training/processed across archive extraction",
    )

    args = parser.parse_args()
    fetch_dataset(
        repo_id=args.repo_id,
        dest=args.dest,
        token=args.token,
        revision=args.revision,
        force=args.force,
        preserve_local_csvs=args.preserve_local_csvs,
    )


if __name__ == "__main__":
    main()
