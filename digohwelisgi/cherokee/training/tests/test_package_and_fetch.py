# -*- coding: utf-8 -*-
"""
Tests for Cherokee dataset packaging and remote fetch script.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from digohwelisgi.cherokee.training.package_dataset import (
    cherokee_target_rel_mapper,
    generate_dataset_card,
    get_cherokee_training_splits,
    resolve_cherokee_audio,
)
from scripts.fetch_training_data import fetch_dataset, is_already_populated


def test_get_cherokee_training_splits():
    repo_root = Path.cwd()
    splits = get_cherokee_training_splits(repo_root)
    assert len(splits) == 6
    names = {s.name for s in splits}
    assert names == {
        "train_orig",
        "valid_orig",
        "test_orig",
        "train_bible",
        "valid_bible",
        "test_bible",
    }
    for s in splits:
        assert s.target_rel_path is not None
        assert str(s.target_rel_path).startswith("training/processed/")


def test_cherokee_target_rel_mapper():
    p1 = "data/training/processed/sentence_audio/Sentence_01.wav"
    res1 = cherokee_target_rel_mapper(p1, Path("/dummy"))
    assert str(res1) == "training/processed/sentence_audio/Sentence_01.wav"

    p2 = "cherokee_new_testament/split_audio/mark_01_01.wav"
    res2 = cherokee_target_rel_mapper(p2, Path("/dummy"))
    assert str(res2) == "projects/cherokee_new_testament/split_audio/mark_01_01.wav"


def test_resolve_cherokee_audio():
    sample_sentence = (
        "data/training/processed/sentence_audio/Sentence_for_entry_1368_01.wav"
    )
    resolved = resolve_cherokee_audio(sample_sentence, Path.cwd())
    assert resolved is not None
    assert resolved.is_file()

    sample_bible = "cherokee_new_testament/split_audio/mark_01_02.wav"
    resolved_bible = resolve_cherokee_audio(sample_bible, Path.cwd())
    assert resolved_bible is not None
    assert resolved_bible.is_file()

    assert resolve_cherokee_audio("nonexistent/foo/bar.wav", Path.cwd()) is None


def test_generate_dataset_card(tmp_path: Path):
    manifest = {
        "dataset_name": "charliemcvicker/test-dataset",
        "total_files": 100,
        "total_duration_sec": 3600.0,
        "total_bytes": 1024 * 1024 * 50,
        "split_counts": {"train_orig": 80, "valid_orig": 20},
    }
    card_path = generate_dataset_card(manifest, tmp_path)
    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert "language:\n- chr" in content
    assert "**Total Audio Files**: 100" in content
    assert "charliemcvicker/test-dataset" in content


def test_is_already_populated(tmp_path: Path):
    assert is_already_populated(tmp_path) is False

    # Create dummy indicators
    (tmp_path / "training" / "processed").mkdir(parents=True)
    (tmp_path / "projects" / "cherokee_new_testament" / "split_audio").mkdir(
        parents=True
    )
    (tmp_path / "training" / "processed" / "sentence_audio").mkdir()
    (tmp_path / "training" / "processed" / "chr-wav2vec2-train.csv").write_text(
        "path,sentence"
    )
    (tmp_path / "training" / "processed" / "bible-wav2vec2-train.csv").write_text(
        "path,sentence"
    )

    assert is_already_populated(tmp_path) is True


def test_fetch_dataset_skips_when_populated(tmp_path: Path):
    (tmp_path / "training" / "processed").mkdir(parents=True)
    (tmp_path / "projects" / "cherokee_new_testament" / "split_audio").mkdir(
        parents=True
    )
    (tmp_path / "training" / "processed" / "sentence_audio").mkdir()
    (tmp_path / "training" / "processed" / "chr-wav2vec2-train.csv").write_text(
        "path,sentence"
    )
    (tmp_path / "training" / "processed" / "bible-wav2vec2-train.csv").write_text(
        "path,sentence"
    )

    with patch("huggingface_hub.snapshot_download") as mock_snap:
        res = fetch_dataset(dest=tmp_path, force=False)
        assert res == tmp_path.resolve()
        mock_snap.assert_not_called()


def test_fetch_dataset_downloads_archive_when_not_populated(tmp_path: Path):
    dest = tmp_path / "new_dest"
    import tarfile

    dummy_tar = tmp_path / "dummy.tar.gz"
    with tarfile.open(dummy_tar, "w:gz") as tar:
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("sample")
        tar.add(dummy_file, arcname="dummy.txt")

    with patch("huggingface_hub.hf_hub_download") as mock_hub_download:
        mock_hub_download.return_value = str(dummy_tar)
        res = fetch_dataset(repo_id="test/repo", dest=dest, force=False)
        assert res == dest.resolve()
        assert (dest / "dummy.txt").exists()
        mock_hub_download.assert_called_once()
