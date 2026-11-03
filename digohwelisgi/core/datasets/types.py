# -*- coding: utf-8 -*-
"""
digohwelisgi.core.datasets.types

Immutable domain models and protocols for audio dataset packaging and staging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

# Protocol for resolving an audio path string found in a CSV split to an actual disk Path
AudioResolver = Callable[[str, Path], Optional[Path]]


@dataclass(frozen=True)
class DatasetSplit:
    """
    Specification for a single dataset split (e.g. train, valid, test).
    """

    name: str
    csv_path: Path
    audio_column: str = "path"
    text_column: str = "sentence"
    # Target relative path inside the staged output directory (defaults to relative path of csv_path or splits/<name>.csv)
    target_rel_path: Optional[Path] = None


@dataclass(frozen=True)
class PackagedAudioRecord:
    """
    Immutable metadata for a single packaged audio file.
    """

    source_path: Path
    target_rel_path: Path
    size_bytes: int
    sha256: str
    duration_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "target_rel_path": str(self.target_rel_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "duration_sec": self.duration_sec,
        }


@dataclass(frozen=True)
class PackagingManifest:
    """
    Durable summary manifest for a packaged dataset.
    """

    dataset_name: str
    created_at: str
    total_files: int
    total_bytes: int
    total_duration_sec: float
    split_counts: Dict[str, int]
    records: Tuple[PackagedAudioRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "split_counts": self.split_counts,
            "records": [r.to_dict() for r in self.records],
        }


@dataclass(frozen=True)
class PackagingConfig:
    """
    Configuration options for staging audio datasets.
    """

    output_dir: Path
    dataset_name: str = "speech-dataset"
    copy_mode: Literal["copy", "hardlink", "symlink"] = "copy"
    compute_checksums: bool = True
    verify_audio_headers: bool = True
    include_record_details_in_manifest: bool = True


@dataclass(frozen=True)
class PackagingResult:
    """
    Result returned by the packaging process.
    """

    output_dir: Path
    manifest: PackagingManifest
    staged_splits: Tuple[Path, ...]
    missing_files: Tuple[str, ...] = field(default_factory=tuple)
    corrupted_files: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_success(self) -> bool:
        return len(self.missing_files) == 0 and len(self.corrupted_files) == 0
