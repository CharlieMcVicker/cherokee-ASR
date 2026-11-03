# -*- coding: utf-8 -*-
"""
digohwelisgi.cherokee.training.package_dataset

Cherokee-specific dataset packaging driver.
Packages the 6 active Cherokee ASR training splits (chr-wav2vec2-* and bible-wav2vec2-*)
and their exact referenced audio files into a clean staging directory ready for Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import tarfile
from typing import List, Optional

from digohwelisgi.core.datasets import (
    DatasetSplit,
    PackagingConfig,
    PackagingResult,
    package_audio_dataset,
    upload_to_huggingface_hub,
)

logger = logging.getLogger(__name__)


def resolve_cherokee_audio(raw_path: str, base_dir: Path) -> Optional[Path]:
    """
    Resolve Cherokee audio paths from split CSVs to absolute filesystem paths.
    Handles sentence_audio and Bible split_audio relative locations.
    """
    clean_p = raw_path.strip()
    direct = Path(clean_p)
    if direct.is_file():
        return direct.resolve()

    # Search known data directories relative to repo root or base_dir
    repo_root = Path.cwd()
    search_prefixes = [
        base_dir,
        repo_root,
        repo_root / "data",
        repo_root / "data" / "training" / "processed",
        repo_root / "data" / "projects",
        repo_root / "data" / "projects" / "cherokee_new_testament",
    ]

    for prefix in search_prefixes:
        cand = (prefix / clean_p).resolve()
        if cand.is_file():
            return cand

    # If basename exists in sentence_audio or split_audio
    for sub in [
        repo_root / "data" / "training" / "processed" / "sentence_audio",
        repo_root / "data" / "projects" / "cherokee_new_testament" / "split_audio",
    ]:
        cand = (sub / direct.name).resolve()
        if cand.is_file():
            return cand

    return None


def cherokee_target_rel_mapper(raw_audio: str, resolved_path: Path) -> Path:
    """
    Map Cherokee audio source paths so that snapshot_download(..., local_dir='data')
    recreates the exact directory layout required by train.py:
      data/training/processed/sentence_audio/...
      data/projects/cherokee_new_testament/split_audio/...
    """
    clean = raw_audio.strip()
    if clean.startswith("data/"):
        return Path(clean[len("data/") :])
    if clean.startswith("cherokee_new_testament/"):
        return Path("projects") / clean
    return Path(clean)


def get_cherokee_training_splits(repo_root: Path) -> List[DatasetSplit]:
    """
    Return the 6 canonical Cherokee ASR training splits.
    """
    data_dir = repo_root / "data" / "training" / "processed"
    splits = [
        DatasetSplit(
            name="train_orig",
            csv_path=data_dir / "chr-wav2vec2-train.csv",
            target_rel_path=Path("training") / "processed" / "chr-wav2vec2-train.csv",
        ),
        DatasetSplit(
            name="valid_orig",
            csv_path=data_dir / "chr-wav2vec2-valid.csv",
            target_rel_path=Path("training") / "processed" / "chr-wav2vec2-valid.csv",
        ),
        DatasetSplit(
            name="test_orig",
            csv_path=data_dir / "chr-wav2vec2-test.csv",
            target_rel_path=Path("training") / "processed" / "chr-wav2vec2-test.csv",
        ),
        DatasetSplit(
            name="train_bible",
            csv_path=data_dir / "bible-wav2vec2-train.csv",
            target_rel_path=Path("training") / "processed" / "bible-wav2vec2-train.csv",
        ),
        DatasetSplit(
            name="valid_bible",
            csv_path=data_dir / "bible-wav2vec2-valid.csv",
            target_rel_path=Path("training") / "processed" / "bible-wav2vec2-valid.csv",
        ),
        DatasetSplit(
            name="test_bible",
            csv_path=data_dir / "bible-wav2vec2-test.csv",
            target_rel_path=Path("training") / "processed" / "bible-wav2vec2-test.csv",
        ),
    ]
    return splits


def generate_dataset_card(manifest_dict: dict, output_dir: Path) -> Path:
    """
    Generate a README.md Dataset Card for the Hugging Face repository.
    """
    card_content = f"""---
language:
- chr
tags:
- audio
- speech-recognition
- cherokee
- wav2vec2
license: mit
size_categories:
- 1K<n<10K
---

# Cherokee ASR Training Dataset (`cherokee-asr-data`)

Production multi-domain Cherokee speech recognition dataset prepared for Wav2Vec2 CTC fine-tuning.

## Dataset Summary
- **Language**: Cherokee (`chr`)
- **Phonetic Orthography**: Canonical T/TH phonetic targets (decoupled tone and vowel lengths)
- **Audio Format**: 16kHz Mono PCM WAV
- **Total Audio Files**: {manifest_dict.get('total_files')}
- **Total Duration**: {round(manifest_dict.get('total_duration_sec', 0) / 3600.0, 2)} hours ({round(manifest_dict.get('total_duration_sec', 0), 1)}s)
- **Total Disk Size**: {round(manifest_dict.get('total_bytes', 0) / (1024 * 1024), 1)} MB

## Splits & Domains
The dataset is partitioned into 6 splits across two acoustic domains:
1. **Sentence Audio (Durbin Feeling corpus)**:
   - `train_orig`: {manifest_dict.get('split_counts', {}).get('train_orig', 0)} samples
   - `valid_orig`: {manifest_dict.get('split_counts', {}).get('valid_orig', 0)} samples
   - `test_orig`: {manifest_dict.get('split_counts', {}).get('test_orig', 0)} samples
2. **Scripture Audio (Cherokee New Testament continuous chapter alignment)**:
   - `train_bible`: {manifest_dict.get('split_counts', {}).get('train_bible', 0)} samples
   - `valid_bible`: {manifest_dict.get('split_counts', {}).get('valid_bible', 0)} samples
   - `test_bible`: {manifest_dict.get('split_counts', {}).get('test_bible', 0)} samples

## Directory Layout
```
├── training/
│   └── processed/
│       ├── chr-wav2vec2-train.csv
│       ├── chr-wav2vec2-valid.csv
│       ├── chr-wav2vec2-test.csv
│       ├── bible-wav2vec2-train.csv
│       ├── bible-wav2vec2-valid.csv
│       ├── bible-wav2vec2-test.csv
│       └── sentence_audio/
└── projects/
    └── cherokee_new_testament/
        └── split_audio/
```

## Quick Start / Fetching
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{manifest_dict.get('dataset_name')}",
    repo_type="dataset",
    local_dir="data",
)
```
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(card_content, encoding="utf-8")
    return readme_path


def create_dataset_tar_archive(
    staged_dir: Path, archive_name: str = "cherokee_asr_data.tar.gz"
) -> Path:
    """
    Compress staged training and projects directories into a single tar.gz archive
    for fast single-request remote downloads (avoiding HF API rate limits).
    """
    tar_path = staged_dir / archive_name
    logger.info("Creating compressed archive: %s...", tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for folder_name in ["training", "projects"]:
            folder = staged_dir / folder_name
            if folder.exists():
                tar.add(folder, arcname=folder_name)
    logger.info(
        "Created archive %s (%.1f MB)",
        tar_path.name,
        tar_path.stat().st_size / (1024 * 1024),
    )
    return tar_path


def package_cherokee_dataset(
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    copy_mode: str = "copy",
    push_to_hub: bool = False,
    repo_id: str = "charliemcvicker/cherokee-asr-data",
    private: bool = True,
    create_archive: bool = True,
) -> PackagingResult:
    """
    Stage and optionally publish the Cherokee ASR training dataset.
    """
    root = (repo_root or Path.cwd()).resolve()
    out = (output_dir or root / "data" / "staging" / "hf_cherokee_asr").resolve()
    splits = get_cherokee_training_splits(root)

    config = PackagingConfig(
        output_dir=out,
        dataset_name=repo_id,
        copy_mode=copy_mode,  # type: ignore
        compute_checksums=True,
        verify_audio_headers=True,
    )

    logger.info("Packaging Cherokee dataset into %s...", out)
    result = package_audio_dataset(
        splits=splits,
        config=config,
        resolver=resolve_cherokee_audio,
        target_rel_mapper=cherokee_target_rel_mapper,
    )

    if not result.is_success:
        logger.error(
            "Packaging completed with errors: %d missing files, %d corrupted files",
            len(result.missing_files),
            len(result.corrupted_files),
        )
        return result

    generate_dataset_card(result.manifest.to_dict(), out)
    if create_archive:
        create_dataset_tar_archive(out)

    logger.info(
        "Staged %d audio files (%.1f MB, %.2f hours) successfully.",
        result.manifest.total_files,
        result.manifest.total_bytes / (1024 * 1024),
        result.manifest.total_duration_sec / 3600.0,
    )

    if push_to_hub:
        logger.info("Uploading dataset to Hugging Face Hub: %s...", repo_id)
        import huggingface_hub

        api = huggingface_hub.HfApi()
        tar_file = out / "cherokee_asr_data.tar.gz"
        if tar_file.exists():
            logger.info(
                "Uploading archive %s via single LFS file upload...", tar_file.name
            )
            api.upload_file(
                path_or_fileobj=str(tar_file),
                path_in_repo="cherokee_asr_data.tar.gz",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add cherokee_asr_data.tar.gz for fast single-request download",
            )
        # Also upload README and manifest
        api.upload_file(
            path_or_fileobj=str(out / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card README.md",
        )
        api.upload_file(
            path_or_fileobj=str(out / "dataset_manifest.json"),
            path_in_repo="dataset_manifest.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset_manifest.json",
        )
        logger.info("Uploaded to https://huggingface.co/datasets/%s", repo_id)

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Package and stage Cherokee ASR dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/staging/hf_cherokee_asr"),
        help="Staging destination folder",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to stage audio files",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload staged dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="charliemcvicker/cherokee-asr-data",
        help="Hugging Face repo ID",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Set repo to public (default: private)",
    )

    args = parser.parse_args()
    result = package_cherokee_dataset(
        output_dir=args.output_dir,
        copy_mode=args.copy_mode,
        push_to_hub=args.push,
        repo_id=args.repo_id,
        private=not args.public,
    )

    if not result.is_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
