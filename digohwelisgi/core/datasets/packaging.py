# -*- coding: utf-8 -*-
"""
digohwelisgi.core.datasets.packaging

Language-agnostic engine for packaging and staging audio datasets.
Extracts only audio files referenced in dataset splits, verifies audio integrity,
computes SHA256 hashes, generates manifests, and stages a clean payload.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import soundfile as sf

from digohwelisgi.core.datasets.types import (
    AudioResolver,
    DatasetSplit,
    PackagedAudioRecord,
    PackagingConfig,
    PackagingManifest,
    PackagingResult,
)

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path, block_size: int = 65536) -> str:
    """
    Compute hex SHA256 checksum for a file.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def default_audio_resolver(raw_path: str, base_dir: Path) -> Optional[Path]:
    """
    Default resolver checking direct path, base_dir relative, and basename.
    """
    clean_p = raw_path.strip()
    direct = Path(clean_p)
    if direct.is_file():
        return direct.resolve()

    cand = (base_dir / clean_p).resolve()
    if cand.is_file():
        return cand

    base_cand = (base_dir / direct.name).resolve()
    if base_cand.is_file():
        return base_cand

    return None


def inspect_audio_file(path: Path) -> Tuple[bool, Optional[float]]:
    """
    Validate that an audio file has a readable sound header and return its duration.
    """
    try:
        info = sf.info(str(path))
        return True, float(info.duration)
    except Exception as exc:
        logger.warning("Corrupted audio file %s: %s", path, exc)
        return False, None


def package_audio_dataset(
    splits: Sequence[DatasetSplit],
    config: PackagingConfig,
    resolver: Optional[AudioResolver] = None,
    target_rel_mapper: Optional[Callable[[str, Path], Path]] = None,
) -> PackagingResult:
    """
    Pure procedure to stage an audio dataset containing only referenced files.

    Args:
        splits: Sequence of split specifications.
        config: Packaging and output staging configuration.
        resolver: Optional custom path resolver. Defaults to default_audio_resolver.

    Returns:
        PackagingResult containing manifest, missing files, and staged split paths.
    """
    resolve_fn = resolver or default_audio_resolver
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scan all splits and resolve audio paths
    split_counts: Dict[str, int] = {}
    missing_files: List[str] = []
    # Map from resolved absolute source path -> target relative path
    unique_sources: Dict[Path, Path] = {}

    for split in splits:
        if not split.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {split.csv_path}")

        count = 0
        with open(split.csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or split.audio_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{split.audio_column}' missing from {split.csv_path}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                raw_audio = row.get(split.audio_column, "").strip()
                if not raw_audio:
                    continue
                count += 1
                resolved = resolve_fn(raw_audio, split.csv_path.parent)
                if resolved is None or not resolved.is_file():
                    missing_files.append(raw_audio)
                else:
                    if resolved not in unique_sources:
                        # Determine target relative path
                        if target_rel_mapper is not None:
                            target_rel = target_rel_mapper(raw_audio, resolved)
                        else:
                            # If raw_audio is relative and doesn't start with root /, preserve its relative structure
                            raw_path_obj = Path(raw_audio)
                            if not raw_path_obj.is_absolute():
                                target_rel = raw_path_obj
                            else:
                                target_rel = Path("audio") / raw_path_obj.name
                        unique_sources[resolved] = target_rel

        split_counts[split.name] = count

    # 2. Inspect audio, stage files, and compute records
    records: List[PackagedAudioRecord] = []
    corrupted_files: List[str] = []
    total_duration = 0.0
    total_bytes = 0

    for source_path, target_rel in unique_sources.items():
        is_valid = True
        duration: Optional[float] = None
        if config.verify_audio_headers:
            is_valid, duration = inspect_audio_file(source_path)
            if not is_valid:
                corrupted_files.append(str(source_path))

        dest_path = output_dir / target_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not dest_path.exists():
            if config.copy_mode == "hardlink":
                os.link(source_path, dest_path)
            elif config.copy_mode == "symlink":
                os.symlink(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)

        size_bytes = dest_path.stat().st_size
        sha256_hash = compute_file_sha256(dest_path) if config.compute_checksums else ""

        record = PackagedAudioRecord(
            source_path=source_path,
            target_rel_path=target_rel,
            size_bytes=size_bytes,
            sha256=sha256_hash,
            duration_sec=duration,
        )
        records.append(record)
        total_bytes += size_bytes
        if duration is not None:
            total_duration += duration

    # 3. Stage split CSV files
    staged_splits: List[Path] = []
    for split in splits:
        target_csv_rel = (
            split.target_rel_path
            if split.target_rel_path is not None
            else Path("splits") / split.csv_path.name
        )
        target_csv_path = output_dir / target_csv_rel
        target_csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split.csv_path, target_csv_path)
        staged_splits.append(target_csv_path)

    # 4. Generate Packaging Manifest
    manifest = PackagingManifest(
        dataset_name=config.dataset_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_files=len(records),
        total_bytes=total_bytes,
        total_duration_sec=total_duration,
        split_counts=split_counts,
        records=(
            tuple(records) if config.include_record_details_in_manifest else tuple()
        ),
    )

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

    return PackagingResult(
        output_dir=output_dir,
        manifest=manifest,
        staged_splits=tuple(staged_splits),
        missing_files=tuple(missing_files),
        corrupted_files=tuple(corrupted_files),
    )


def upload_to_huggingface_hub(
    staged_dir: Path,
    repo_id: str,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
) -> str:
    """
    Upload a staged dataset directory directly to Hugging Face Hub using huggingface_hub.

    Args:
        staged_dir: Local staged directory containing audio, splits, and manifest.
        repo_id: Target Hugging Face repository (e.g. 'charliemcvicker/cherokee-asr-data').
        repo_type: Repository type ('dataset' or 'model').
        token: Optional Hugging Face auth token (defaults to environment or cached token).
        private: Whether to ensure repository is private if created.
        commit_message: Optional commit message for the upload.

    Returns:
        The URL of the Hugging Face repository.
    """
    import huggingface_hub

    api = huggingface_hub.HfApi(token=token)

    # Create repo if not existing
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    msg = commit_message or f"Upload staged dataset: {staged_dir.name}"
    api.upload_folder(
        folder_path=str(staged_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
    )

    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Uploaded dataset successfully to %s", repo_url)
    return repo_url
