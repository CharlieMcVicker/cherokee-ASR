# -*- coding: utf-8 -*-
"""
Tests for digohwelisgi.core.datasets.packaging
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

from digohwelisgi.core.datasets import (
    DatasetSplit,
    PackagingConfig,
    compute_file_sha256,
    default_audio_resolver,
    inspect_audio_file,
    package_audio_dataset,
    upload_to_huggingface_hub,
)


def _create_synthetic_wav(
    path: Path, duration_sec: float = 0.5, sr: int = 16000
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(duration_sec * sr)
    audio = np.sin(
        2 * np.pi * 440 * np.linspace(0, duration_sec, samples, endpoint=False)
    ).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


def test_inspect_audio_and_sha256(tmp_path: Path):
    wav_path = _create_synthetic_wav(tmp_path / "test.wav", duration_sec=1.0)
    is_valid, duration = inspect_audio_file(wav_path)
    assert is_valid is True
    assert duration is not None
    assert round(duration, 1) == 1.0

    sha = compute_file_sha256(wav_path)
    assert isinstance(sha, str)
    assert len(sha) == 64

    # Test corrupted file
    corrupted_path = tmp_path / "corrupted.wav"
    corrupted_path.write_bytes(b"not a valid wav file header")
    is_valid, duration = inspect_audio_file(corrupted_path)
    assert is_valid is False
    assert duration is None


def test_package_audio_dataset_isolates_referenced_files_only(tmp_path: Path):
    src_dir = tmp_path / "source"
    src_dir.mkdir()

    # Create 3 wav files
    wav1 = _create_synthetic_wav(src_dir / "audio1.wav", duration_sec=0.5)
    wav2 = _create_synthetic_wav(src_dir / "audio2.wav", duration_sec=1.0)
    wav3 = _create_synthetic_wav(
        src_dir / "audio3.wav", duration_sec=1.5
    )  # unreferenced

    # Create 2 split CSVs referencing only wav1 and wav2
    train_csv = src_dir / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])
        writer.writerow(["audio1.wav", "first sentence"])

    valid_csv = src_dir / "valid.csv"
    with open(valid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])
        writer.writerow(["audio2.wav", "second sentence"])

    out_dir = tmp_path / "staged"
    splits = [
        DatasetSplit(name="train", csv_path=train_csv),
        DatasetSplit(name="valid", csv_path=valid_csv),
    ]
    config = PackagingConfig(
        output_dir=out_dir,
        dataset_name="test-corpus",
        copy_mode="copy",
    )

    result = package_audio_dataset(splits=splits, config=config)

    assert result.is_success is True
    assert len(result.missing_files) == 0
    assert len(result.corrupted_files) == 0
    assert result.manifest.total_files == 2
    assert result.manifest.split_counts == {"train": 1, "valid": 1}

    # Verify only wav1 and wav2 are staged, and wav3 is absent
    assert (out_dir / "audio1.wav").exists()
    assert (out_dir / "audio2.wav").exists()
    assert not (out_dir / "audio3.wav").exists()

    # Verify manifest exists and is readable
    manifest_file = out_dir / "dataset_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_files"] == 2
        assert len(data["records"]) == 2


def test_package_audio_dataset_detects_missing_files(tmp_path: Path):
    src_dir = tmp_path / "source"
    src_dir.mkdir()

    wav1 = _create_synthetic_wav(src_dir / "audio1.wav", duration_sec=0.5)

    train_csv = src_dir / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])
        writer.writerow(["audio1.wav", "exists"])
        writer.writerow(["nonexistent_audio.wav", "missing"])

    splits = [DatasetSplit(name="train", csv_path=train_csv)]
    config = PackagingConfig(output_dir=tmp_path / "staged")

    result = package_audio_dataset(splits=splits, config=config)

    assert result.is_success is False
    assert "nonexistent_audio.wav" in result.missing_files
    assert result.manifest.total_files == 1


def test_upload_to_huggingface_hub(tmp_path: Path):
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "manifest.json").write_text("{}")

    with patch("huggingface_hub.HfApi") as mock_hf_api_cls:
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        url = upload_to_huggingface_hub(
            staged_dir=staged_dir,
            repo_id="test-user/test-dataset",
            token="hf_fake_token",
            private=True,
        )

        assert url == "https://huggingface.co/datasets/test-user/test-dataset"
        mock_api.create_repo.assert_called_once_with(
            repo_id="test-user/test-dataset",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        mock_api.upload_folder.assert_called_once()
