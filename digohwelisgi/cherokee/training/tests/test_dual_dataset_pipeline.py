# -*- coding: utf-8 -*-
"""
test_dual_dataset_pipeline.py

Unit and integration tests for Phase 3 Dual-Stream Dataset Preparation & Vocabulary Serialization.
Includes Integration Gate 3 dataset integrity audit across all 6 CSV splits.
"""

import csv
import json
from pathlib import Path
import tarfile
import pytest

from digohwelisgi.cherokee.orthography import (
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
    decode_cv_ids,
    decode_tone_ids,
    encode_cv_text,
    encode_tone_tokens,
    reconcile_dual_stream,
    save_dual_vocabularies,
    to_moraic_tone_tokens,
)
from scripts.fetch_training_data import fetch_dataset


def test_vocab_tokenizers_bidirectional(tmp_path: Path) -> None:
    """
    Encode and decode arbitrary CV sentences and Tone token sequences;
    assert exact recovery and verify serialized vocab sizes (19 and 10).
    """
    vocab_cv_path, vocab_tone_path = save_dual_vocabularies(tmp_path)
    with open(vocab_cv_path, encoding="utf-8") as f:
        v_cv = json.load(f)
    with open(vocab_tone_path, encoding="utf-8") as f:
        v_tone = json.load(f)

    assert len(v_cv) == 19
    assert len(v_tone) == 10
    assert "1_1" not in v_tone
    assert "4_1" not in v_tone

    sample_cv, sample_tone = to_moraic_tone_tokens(
        "ná *ani:chű:ja* sgwà:hle̋:sdi dà:ná:ne:lo:hv́sga"
    )
    cv_ids = encode_cv_text(sample_cv, v_cv)
    tone_ids = encode_tone_tokens(sample_tone, v_tone)

    assert v_cv["[UNK]"] not in cv_ids
    assert v_tone["[UNK]"] not in tone_ids

    decoded_cv = decode_cv_ids(cv_ids, v_cv)
    decoded_tone = decode_tone_ids(tone_ids, v_tone)

    assert decoded_cv == sample_cv
    assert decoded_tone == sample_tone


def test_bible_sample_masking() -> None:
    """Verify that Bible rows (empty tone_tokens) yield tone_labels = [-100]."""
    assert encode_tone_tokens("") == [-100]
    assert encode_tone_tokens("   ") == [-100]
    assert encode_tone_tokens(None) == [-100]


def test_dataset_split_sibilant_integrity() -> None:
    """
    Integration Gate 3 Dataset Audit:
    Scan all 6 generated CSV splits on disk and verify:
      - 0 colons ':' in cv_text
      - 0 words start with 'hs-' across all splits
      - 100% of non-empty tone_tokens contain strictly valid tokens from vocab_tone.json
      - 100% of characters in cv_text belong to vocab.json
      - 100% roundtrip match between cv_text + tone_tokens and canonical_toned on chr splits
    """
    processed_dir = Path("data/training/processed")
    chr_splits = [
        processed_dir / "chr-wav2vec2-train.csv",
        processed_dir / "chr-wav2vec2-valid.csv",
        processed_dir / "chr-wav2vec2-test.csv",
    ]
    bible_splits = [
        processed_dir / "bible-wav2vec2-train.csv",
        processed_dir / "bible-wav2vec2-valid.csv",
        processed_dir / "bible-wav2vec2-test.csv",
    ]

    valid_cv_chars = set(SEGMENTAL_CV_VOCAB.keys()) | {" "} - {"[PAD]", "[UNK]", "|"}
    valid_tone_toks = set(MORAIC_TONE_VOCAB.keys()) - {"[PAD]", "[UNK]"}

    for csv_path in chr_splits:
        assert csv_path.exists(), f"Missing split: {csv_path}"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            cv_text = row["cv_text"]
            tone_tokens = row["tone_tokens"]
            canonical_toned = row["canonical_toned"]

            assert ":" not in cv_text
            for word in cv_text.split():
                assert not word.startswith("hs"), f"Leading hs- in {csv_path}: {word}"
            assert set(cv_text).issubset(valid_cv_chars)

            toks = tone_tokens.split()
            assert len(toks) > 0
            assert set(toks).issubset(valid_tone_toks)
            assert (
                reconcile_dual_stream(cv_text, tone_tokens, mode="dp")
                == canonical_toned
            )

    for csv_path in bible_splits:
        assert csv_path.exists(), f"Missing split: {csv_path}"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            cv_text = row["cv_text"]
            tone_tokens = row["tone_tokens"]

            assert ":" not in cv_text
            for word in cv_text.split():
                assert not word.startswith("hs"), f"Leading hs- in {csv_path}: {word}"
            assert set(cv_text).issubset(valid_cv_chars)
            assert tone_tokens == ""
            assert encode_tone_tokens(tone_tokens) == [-100]


def test_fetch_dataset_preserve_local_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify --preserve-local-csvs prevents archive extraction from overwriting local CSVs."""
    dest = tmp_path / "data"
    proc_dir = dest / "training" / "processed"
    proc_dir.mkdir(parents=True)
    local_csv = proc_dir / "chr-wav2vec2-train.csv"
    local_csv.write_text(
        "path,cv_text,tone_tokens\nlocal.wav,sakwu,C 2_1\n", encoding="utf-8"
    )

    # Create a mock tar.gz archive that contains an older version of chr-wav2vec2-train.csv
    archive_src = tmp_path / "archive_src"
    (archive_src / "training" / "processed").mkdir(parents=True)
    (archive_src / "training" / "processed" / "chr-wav2vec2-train.csv").write_text(
        "path,sentence\nold.wav,hsakwu\n", encoding="utf-8"
    )
    archive_path = tmp_path / "cherokee_asr_data.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(archive_src / "training", arcname="training")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: str(archive_path),
    )

    fetch_dataset(dest=dest, force=True, preserve_local_csvs=True)
    assert (
        local_csv.read_text(encoding="utf-8")
        == "path,cv_text,tone_tokens\nlocal.wav,sakwu,C 2_1\n"
    )
