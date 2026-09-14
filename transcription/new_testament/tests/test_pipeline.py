# -*- coding: utf-8 -*-
"""
Unit tests for New Testament pipeline in transcription.new_testament.pipeline.
"""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch
import numpy as np
from pydub import AudioSegment
import pytest
import torch

from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.distance_metrics import (
    ConfusionMatrixCostMetric,
    DefaultCERDistanceMetric,
)
from transcription.alignment.extractors import (
    CachedASREmissionsExtractor,
    PrecomputedEmissionsExtractor,
)
from transcription.alignment.models import AlignmentOutput, TextChunk, TokenEmission
from transcription.audio.segment import AudioChunk
from transcription.new_testament.pipeline import (
    align_chapter,
    load_chapter_transcript,
    reconcile_syllabary_asr,
)


class MockASRModel:
    def __init__(self):
        self.processor = MagicMock()
        self.processor.tokenizer.pad_token_id = 0
        self.processor.tokenizer.get_vocab.return_value = {
            "[PAD]": 0,
            "a": 1,
            "e": 2,
            "i": 3,
            "o": 4,
            "u": 5,
            "v": 6,
            "d": 7,
            "l": 8,
            "n": 9,
            "s": 10,
            "g": 11,
            "y": 12,
            "k": 13,
            "h": 14,
        }
        self.model_name = "mock_model"
        self.call_count = 0

    def get_logits(self, samples: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        self.call_count += 1
        n_frames = max(200, len(samples) // 320)
        vocab_size = 15
        logits = torch.zeros((n_frames, vocab_size), dtype=torch.float32)
        logits[:, 1] = 2.0  # boost token 'a'
        return logits

    def decode(self, lpz: np.ndarray):
        res = MagicMock()
        res.confidence = 0.95
        res.text = "adalenisgv yisdv"
        return res


@pytest.fixture
def dummy_audio_path(tmp_path: Path) -> Path:
    audio_path = tmp_path / "dummy_audio.wav"
    silence = AudioSegment.silent(duration=5000, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


@pytest.fixture
def dummy_transcript(tmp_path: Path) -> Path:
    t_path = tmp_path / "transcript.json"
    data = {
        "020101": {
            "image_path": "images/020101.png",
            "english": "The beginning of the gospel",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
        },
        "020102": {
            "image_path": "images/020102.png",
            "english": "As it is written",
            "cherokee": "ᎾᏍᎩᏯ ᎯᎠ ᏥᏂᎬᏅ",
            "phonetic": "Na-s-gi-ya hi-a tsi-ni-gv-nv",
        },
    }
    t_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return t_path


@pytest.fixture
def dummy_cost_matrix(tmp_path: Path) -> Path:
    matrix_path = tmp_path / "confusion_cost_matrix_prebible.json"
    data = {
        "unigram_costs": {
            "a": {"a": 0.0, "e": 0.5},
            "e": {"e": 0.0, "a": 0.5},
        },
        "insertion_cost": 1.0,
        "deletion_cost": 1.0,
        "default_substitution_cost": 0.9,
    }
    matrix_path.write_text(json.dumps(data), encoding="utf-8")
    return matrix_path


def test_load_chapter_transcript(dummy_transcript: Path):
    data = load_chapter_transcript(dummy_transcript)
    assert "020101" in data
    assert data["020101"]["cherokee"] == "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ"

    with pytest.raises(FileNotFoundError):
        load_chapter_transcript("nonexistent_path_xyz.json")


def test_reconcile_syllabary_asr():
    enriched, pairs = reconcile_syllabary_asr("ᎣᏏᏲ", "osiyo")
    assert isinstance(enriched, str)
    assert len(pairs) > 0


def test_align_chapter_with_custom_distance_metric_and_extractor(
    tmp_path: Path, dummy_transcript: Path
):
    dummy_emissions = [
        TokenEmission(word="adalenisgv", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="yisdv", start_sec=0.7, end_sec=1.1, confidence=0.92),
        TokenEmission(word="kanohedv", start_sec=1.2, end_sec=1.8, confidence=0.90),
    ]
    extractor = PrecomputedEmissionsExtractor(token_emissions=dummy_emissions)
    metric = DefaultCERDistanceMetric()

    out_dir = tmp_path / "output_custom"

    res = align_chapter(
        audio_path="dummy_audio.wav",
        transcript_path=dummy_transcript,
        output_dir=out_dir,
        export_praat=True,
        reconcile=True,
        distance_metric=metric,
        emissions_extractor=extractor,
    )

    assert isinstance(res, AlignmentOutput)
    assert len(res.aligned_chunks) >= 1
    assert res.aligned_chunks[0].chunk_id == "020101"
    assert len(res.aligned_chunks[0].words) > 0
    assert (out_dir / "alignment.TextGrid").exists()
    assert (out_dir / "alignment_manifest.json").exists()


def test_align_chapter_with_cached_extractor_and_confusion_metric(
    tmp_path: Path, dummy_transcript: Path, dummy_cost_matrix: Path
):
    dummy_emissions = [
        TokenEmission(word="adalenisgv", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="yisdv", start_sec=0.7, end_sec=1.1, confidence=0.92),
        TokenEmission(word="kanohedv", start_sec=1.2, end_sec=1.8, confidence=0.90),
    ]
    base_extractor = PrecomputedEmissionsExtractor(token_emissions=dummy_emissions)
    cached_extractor = CachedASREmissionsExtractor(
        extractor=base_extractor,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="test_prefix",
    )
    metric = ConfusionMatrixCostMetric.from_json(dummy_cost_matrix)

    out_dir = tmp_path / "output_cached"

    res = align_chapter(
        audio_path="dummy_audio.wav",
        transcript_path=dummy_transcript,
        output_dir=out_dir,
        distance_metric=metric,
        emissions_extractor=cached_extractor,
    )

    assert isinstance(res, AlignmentOutput)
    assert len(res.aligned_chunks) >= 1
    assert res.metrics is not None
    assert res.metrics.matched_chunks >= 1


def test_align_chapter_ctc_segmentation_with_mock_aligner(
    tmp_path: Path, dummy_audio_path: Path, dummy_transcript: Path
):
    mock_model = MockASRModel()
    ctc_aligner = CTCSegmentationAligner(
        model=cast(Any, mock_model),
        cache=True,
        cache_dir=tmp_path / "ctc_cache",
    )

    out_dir = tmp_path / "output_ctc"

    res = align_chapter(
        audio_path=dummy_audio_path,
        transcript_path=dummy_transcript,
        output_dir=out_dir,
        export_praat=True,
        export_manifest=True,
        debug_export=True,
        reconcile=True,
        engine="ctc",
        ctc_aligner=ctc_aligner,
    )

    assert isinstance(res, AlignmentOutput)
    assert len(res.aligned_chunks) == 2
    assert res.aligned_chunks[0].chunk_id == "020101"
    assert res.aligned_chunks[1].chunk_id == "020102"

    # Verify 4-tier Praat TextGrid
    tg_path = out_dir / "alignment.TextGrid"
    assert tg_path.exists()
    tg_content = tg_path.read_text(encoding="utf-8")
    assert 'name = "Chunks"' in tg_content
    assert 'name = "Words"' in tg_content
    assert 'name = "Padded Words"' in tg_content
    assert 'name = "Reconciled Transcriptions"' in tg_content
    assert "size = 4" in tg_content

    # Verify manifest JSON
    manifest_path = out_dir / "alignment_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest_data["lines"]) == 2
    assert "reconciled_words" in manifest_data["lines"][0]
    assert manifest_data["lines"][0]["line_id"] == "020101"

    # Verify debug JSON
    debug_path = out_dir / "alignment_debug.json"
    assert debug_path.exists()


def test_align_chapter_default_fallback_instantiation(
    tmp_path: Path, dummy_audio_path: Path, dummy_transcript: Path
):
    out_dir = tmp_path / "output_default"

    with patch(
        "transcription.models.asr_model.CherokeeASRModel.from_pretrained_or_best"
    ) as mock_model_load:
        mock_model = MockASRModel()
        mock_model_load.return_value = mock_model

        res = align_chapter(
            audio_path=dummy_audio_path,
            transcript_path=dummy_transcript,
            output_dir=out_dir,
            cache_dir=tmp_path / "cache_dir",
            model_revision="5464d15",
            engine="ctc",
        )

        assert mock_model_load.called
        assert isinstance(res, AlignmentOutput)
        assert len(res.aligned_chunks) == 2


def test_realign_book_single_chapter(tmp_path: Path, monkeypatch):
    import scripts.realign_bible as rb

    mock_model = MockASRModel()
    ctc_aligner = CTCSegmentationAligner(
        model=cast(Any, mock_model),
        cache=False,
    )

    # Mock paths
    split_dir = tmp_path / "split_audio"
    alignments_dir = tmp_path / "alignments"
    train_csvs_dir = tmp_path / "train_csvs"
    audio_src_dir = tmp_path / "audio_source"
    transcripts_dir = tmp_path / "book_transcripts"

    for d in [
        split_dir,
        alignments_dir,
        train_csvs_dir,
        audio_src_dir,
        transcripts_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rb, "SPLIT_AUDIO_DIR", split_dir)
    monkeypatch.setattr(rb, "ALIGNMENTS_DIR", alignments_dir)
    monkeypatch.setattr(rb, "TRAIN_CSVS_DIR", train_csvs_dir)
    monkeypatch.setattr(rb, "AUDIO_SRC_DIR", audio_src_dir)
    monkeypatch.setattr(rb, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(rb, "PRAAT_OUT_DIR", tmp_path / "praat_out")

    # Create dummy chapter 1 audio and transcript
    silence = AudioSegment.silent(duration=3000, frame_rate=16000)
    silence.export(str(audio_src_dir / "mark_01.mp3"), format="mp3")

    transcript_data = {
        "020101": {
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
            "english": "The beginning of the gospel",
        },
    }
    with open(transcripts_dir / "mark_01.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f)

    records, csv_rows = rb.realign_book(
        book="mark",
        chapter=1,
        ctc_aligner=ctc_aligner,
        cache=False,
    )

    assert len(records) == 1
    assert records[0]["verse_id"] == "020101"
    assert records[0]["chapter"] == 1
    assert "has_anomalies" in records[0]
    assert (alignments_dir / "mark_alignment_records.json").exists()
    assert (train_csvs_dir / "mark.csv").exists()
    assert (split_dir / "mark_01_01.wav").exists()


def test_realign_book_flags_and_excludes_anomaly_verse_from_train_csv(
    tmp_path: Path, monkeypatch
):
    """
    Test that when an aligned chunk has flagged words (e.g. transcript typo/low confidence),
    the verse is marked with has_anomalies: True in alignment records and is NOT added
    to the generated training CSV (mark.csv).
    """
    import csv
    import scripts.realign_bible as rb
    from transcription.alignment.models import (
        AlignedChunk,
        AlignmentMetrics,
        AlignmentOutput,
        WordInterval,
    )

    split_dir = tmp_path / "split_audio"
    alignments_dir = tmp_path / "alignments"
    train_csvs_dir = tmp_path / "train_csvs"
    audio_src_dir = tmp_path / "audio_source"
    transcripts_dir = tmp_path / "book_transcripts"

    for d in [
        split_dir,
        alignments_dir,
        train_csvs_dir,
        audio_src_dir,
        transcripts_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rb, "SPLIT_AUDIO_DIR", split_dir)
    monkeypatch.setattr(rb, "ALIGNMENTS_DIR", alignments_dir)
    monkeypatch.setattr(rb, "TRAIN_CSVS_DIR", train_csvs_dir)
    monkeypatch.setattr(rb, "AUDIO_SRC_DIR", audio_src_dir)
    monkeypatch.setattr(rb, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(rb, "PRAAT_OUT_DIR", tmp_path / "praat_out")

    silence = AudioSegment.silent(duration=4000, frame_rate=16000)
    silence.export(str(audio_src_dir / "mark_01.mp3"), format="mp3")

    transcript_data = {
        "020101": {
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
            "english": "The beginning of the gospel",
        },
        "020102": {
            "cherokee": "ᎾᏍᎩᏯ ᎯᎠ ᏥᏂᎬᏅ",
            "phonetic": "Na-s-gi-ya hi-a tsi-ni-gv-nv",
            "english": "As it is written",
        },
    }
    with open(transcripts_dir / "mark_01.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f)

    # Mock align_chapter return: Verse 020101 has flagged typo word, Verse 020102 is clean
    chunk1 = AlignedChunk(
        chunk_id="020101",
        start_sec=0.1,
        end_sec=1.5,
        words=[
            WordInterval(
                word="adalenisgv",
                start_sec=0.1,
                end_sec=0.5,
                confidence=0.9,
                flagged=False,
                emitted_word="adalenisgv",
            ),
            WordInterval(
                word="yisdv",
                start_sec=0.5,
                end_sec=0.9,
                confidence=0.005,
                flagged=True,
                emitted_word="ysdv",
            ),  # Typo/anomaly!
            WordInterval(
                word="kanohedv",
                start_sec=0.9,
                end_sec=1.5,
                confidence=0.88,
                flagged=False,
                emitted_word="kanohedv",
            ),
        ],
        distance_score=0.1,
        emitted_text="adalenisgv ysdv kanohedv",
    )
    chunk2 = AlignedChunk(
        chunk_id="020102",
        start_sec=1.6,
        end_sec=3.0,
        words=[
            WordInterval(
                word="nasgiya",
                start_sec=1.6,
                end_sec=2.0,
                confidence=0.95,
                flagged=False,
                emitted_word="nasgiya",
            ),
            WordInterval(
                word="hia",
                start_sec=2.0,
                end_sec=2.4,
                confidence=0.92,
                flagged=False,
                emitted_word="hia",
            ),
            WordInterval(
                word="tsinigvnv",
                start_sec=2.4,
                end_sec=3.0,
                confidence=0.91,
                flagged=False,
                emitted_word="tsinigvnv",
            ),
        ],
        distance_score=0.05,
        emitted_text="nasgiya hia tsinigvnv",
    )

    mock_res = AlignmentOutput(
        aligned_chunks=[chunk1, chunk2],
        source_id="mark_01",
    )

    with patch("scripts.realign_bible.align_chapter", return_value=mock_res):
        records, csv_rows = rb.realign_book(
            book="mark",
            chapter=1,
            cache=False,
        )

    # Check alignment records: both verses recorded, but 020101 marked as anomaly
    assert len(records) == 2
    rec1 = next(r for r in records if r["verse_id"] == "020101")
    rec2 = next(r for r in records if r["verse_id"] == "020102")
    assert rec1["has_anomalies"] is True
    assert rec2["has_anomalies"] is False

    # Check training CSV rows: ONLY clean verse 020102 is present
    assert len(csv_rows) == 1
    assert "mark_01_01.wav" not in csv_rows[0]["path"]
    assert "mark_01_02.wav" in csv_rows[0]["path"]

    # Verify written CSV file
    train_csv_file = train_csvs_dir / "mark.csv"
    with open(train_csv_file, "r", encoding="utf-8") as f:
        rows_on_disk = list(csv.DictReader(f))
    assert len(rows_on_disk) == 1
    assert any("mark_01_02.wav" in r["path"] for r in rows_on_disk)
    assert not any("mark_01_01.wav" in r["path"] for r in rows_on_disk)


def test_realign_book_excludes_verse_with_missing_emitted_text_from_train_csv(
    tmp_path: Path, monkeypatch
):
    """
    Test that when an aligned chunk has no emitted_text (empty or whitespace),
    it is excluded from training CSV export with no fallback to phonetic text.
    """
    import csv
    import scripts.realign_bible as rb
    from transcription.alignment.models import (
        AlignedChunk,
        AlignmentOutput,
        WordInterval,
    )

    split_dir = tmp_path / "split_audio"
    alignments_dir = tmp_path / "alignments"
    train_csvs_dir = tmp_path / "train_csvs"
    audio_src_dir = tmp_path / "audio_source"
    transcripts_dir = tmp_path / "book_transcripts"

    for d in [
        split_dir,
        alignments_dir,
        train_csvs_dir,
        audio_src_dir,
        transcripts_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rb, "SPLIT_AUDIO_DIR", split_dir)
    monkeypatch.setattr(rb, "ALIGNMENTS_DIR", alignments_dir)
    monkeypatch.setattr(rb, "TRAIN_CSVS_DIR", train_csvs_dir)
    monkeypatch.setattr(rb, "AUDIO_SRC_DIR", audio_src_dir)
    monkeypatch.setattr(rb, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(rb, "PRAAT_OUT_DIR", tmp_path / "praat_out")

    silence = AudioSegment.silent(duration=4000, frame_rate=16000)
    silence.export(str(audio_src_dir / "mark_01.mp3"), format="mp3")

    transcript_data = {
        "020101": {
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
            "english": "The beginning of the gospel",
        },
        "020102": {
            "cherokee": "ᎾᏍᎩᏯ ᎯᎠ ᏥᏂᎬᏅ",
            "phonetic": "Na-s-gi-ya hi-a tsi-ni-gv-nv",
            "english": "As it is written",
        },
    }
    with open(transcripts_dir / "mark_01.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f)

    # Chunk 1 has empty emitted_text (unaligned/missing emissions), Chunk 2 has valid emissions
    chunk1 = AlignedChunk(
        chunk_id="020101",
        start_sec=0.0,
        end_sec=1.5,
        words=[],
        distance_score=1.0,
        emitted_text="",
    )
    chunk2 = AlignedChunk(
        chunk_id="020102",
        start_sec=1.6,
        end_sec=3.0,
        words=[
            WordInterval(
                word="nasgiya",
                start_sec=1.6,
                end_sec=2.0,
                confidence=0.95,
                flagged=False,
                emitted_word="nasgiya",
            ),
            WordInterval(
                word="hia",
                start_sec=2.0,
                end_sec=2.4,
                confidence=0.92,
                flagged=False,
                emitted_word="hia",
            ),
            WordInterval(
                word="tsinigvnv",
                start_sec=2.4,
                end_sec=3.0,
                confidence=0.91,
                flagged=False,
                emitted_word="tsinigvnv",
            ),
        ],
        distance_score=0.05,
        emitted_text="nasgiya hia tsinigvnv",
    )

    mock_res = AlignmentOutput(
        aligned_chunks=[chunk1, chunk2],
        source_id="mark_01",
    )

    with patch("scripts.realign_bible.align_chapter", return_value=mock_res):
        records, csv_rows = rb.realign_book(
            book="mark",
            chapter=1,
            cache=False,
        )

    # Both recorded in JSON
    assert len(records) == 2
    rec1 = next(r for r in records if r["verse_id"] == "020101")
    assert rec1["reconciled_phonetics"] == ""

    # Only chunk 2 with non-empty emitted_text is in CSV
    assert len(csv_rows) == 1
    assert "mark_01_02.wav" in csv_rows[0]["path"]
    assert csv_rows[0]["sentence"] == "nasgiya hia tsinigvnv"

    train_csv_file = train_csvs_dir / "mark.csv"
    with open(train_csv_file, "r", encoding="utf-8") as f:
        rows_on_disk = list(csv.DictReader(f))
    assert len(rows_on_disk) == 1
    assert any("mark_01_02.wav" in r["path"] for r in rows_on_disk)
    assert not any("mark_01_01.wav" in r["path"] for r in rows_on_disk)
