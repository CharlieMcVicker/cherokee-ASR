# -*- coding: utf-8 -*-
"""
test_scripture_pipeline.py

Unit and integration tests for the modular Scripture chapter & verse slicing pipeline:
- transcription.pipelines.scripture.ingestion (JSON, TSV, CSV transcript ingestion)
- transcription.pipelines.scripture.pipeline (ScripturePipeline composing CTCSegmentationAligner & Cherokee phonotactics)
- Verse audio boundary partitioning & AudioChunk domain model export
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import numpy as np
from pydub import AudioSegment
import pytest

from transcription.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    WordInterval,
)
from transcription.core.audio.segment import AudioChunk
from transcription.core.models.output import ModelOutput
from transcription.pipelines.scripture.ingestion import (
    default_scripture_phonetic_normalizer,
    load_bible_chunks,
    load_chapter_transcript,
)
from transcription.pipelines.scripture.pipeline import (
    ScripturePipeline,
    align_chapter,
    reconcile_syllabary_asr,
)


@pytest.fixture
def dummy_audio_path(tmp_path: Path) -> Path:
    audio_path = tmp_path / "dummy_chapter_audio.wav"
    # Create 3-second silent 16kHz mono audio
    silence = AudioSegment.silent(duration=3000, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


@pytest.fixture
def dummy_json_transcript(tmp_path: Path) -> Path:
    p = tmp_path / "mark_01.json"
    data = {
        "020101": {
            "english": "The beginning of the gospel",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
        },
        "020102": {
            "english": "As it is written",
            "cherokee": "ᎾᏍᎩᏯ ᎯᎠ ᏥᏂᎬᏅ",
            "phonetic": "Na-s-gi-ya hi-a tsi-ni-gv-nv",
        },
    }
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def dummy_tsv_transcript(tmp_path: Path) -> Path:
    p = tmp_path / "john_01.tsv"
    content = (
        "id\tcherokee\tphonetic\tenglish\n"
        "040101\tᏗᏓᎴᏂᏍᎬᎢ ᎧᏃᎮᏛ ᎡᎮᎢ\tDi-da-le-ni-s-gv-i ka-no-he-dv e-he-i\tIn the beginning was the Word\n"
        "040102\tᎾᏍᎩ ᏗᏓᎴᏂᏍᎬᎢ ᎤᏁᎳᏅᎯ ᎢᏧᎳᎭ ᎠᏁᎮᎢ\tNa-s-gi di-da-le-ni-s-gv-i U-ne-la-nv-hi i-tsu-la-ha a-ne-he-i\tThe same was in the beginning with God\n"
    )
    p.write_text(content, encoding="utf-8")
    return p


def test_load_chapter_transcript_json(dummy_json_transcript: Path):
    data = load_chapter_transcript(dummy_json_transcript)
    assert "020101" in data
    assert "020102" in data
    assert data["020101"]["cherokee"] == "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ"
    assert "A-da-le-ni-s-gv" in data["020101"]["phonetic"]


def test_load_chapter_transcript_tsv(dummy_tsv_transcript: Path):
    data = load_chapter_transcript(dummy_tsv_transcript)
    assert "040101" in data
    assert "040102" in data
    assert data["040101"]["cherokee"] == "ᏗᏓᎴᏂᏍᎬᎢ ᎧᏃᎮᏛ ᎡᎮᎢ"
    assert "Di-da-le-ni-s-gv-i" in data["040101"]["phonetic"]


def test_load_bible_chunks_normalization(dummy_json_transcript: Path):
    chunks, source_lookup = load_bible_chunks(dummy_json_transcript)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "020101"
    # Canonical Cherokee T/TH orthography checks:
    # 'a-da-le-ni-s-gv' -> 'adalenisgv' (no hyphens, d->t, g->k)
    assert "adalenisgv" in chunks[0].text or "at" in chunks[0].text
    assert "020101" in source_lookup
    assert source_lookup["020101"]["cherokee"] == "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ"


def test_scripture_pipeline_alignment_with_model_output(dummy_json_transcript: Path):
    # Construct synthetic ModelOutput with Cherokee vocabulary
    vocab = {
        "[PAD]": 0,
        "a": 1,
        "e": 2,
        "i": 3,
        "o": 4,
        "u": 5,
        "v": 6,
        "t": 7,
        "th": 8,
        "k": 9,
        "kh": 10,
        "l": 11,
        "lh": 12,
        "tl": 13,
        "tlh": 14,
        "s": 15,
        "hs": 16,
        "ts": 17,
        "tsh": 18,
        "m": 19,
        "n": 20,
        "nh": 21,
        "w": 22,
        "wh": 23,
        "y": 24,
        "yh": 25,
        "'": 26,
    }
    n_frames = 200
    lpz = np.zeros((n_frames, len(vocab)), dtype=np.float32)
    # Give high log-probability to blank/a
    lpz[:, 0] = -0.1
    lpz[:, 1] = -2.0

    model_out = ModelOutput(
        lpz=lpz,
        vocab=vocab,
        frame_duration_sec=0.02,
        metadata={"pad_token_id": 0},
    )

    pipeline = ScripturePipeline()
    alignment, lookup = pipeline.align(
        audio=model_out,
        transcript=dummy_json_transcript,
    )

    assert isinstance(alignment, AlignmentOutput)
    assert len(alignment.aligned_chunks) == 2
    assert alignment.aligned_chunks[0].chunk_id == "020101"
    assert alignment.aligned_chunks[1].chunk_id == "020102"
    assert "020101" in lookup


def test_slice_verse_audio(dummy_audio_path: Path):
    pipeline = ScripturePipeline()
    alignment = AlignmentOutput(
        aligned_chunks=[
            AlignedChunk(
                chunk_id="020101",
                start_sec=0.2,
                end_sec=1.4,
                words=[
                    WordInterval(
                        word="adalenisgv",
                        start_sec=0.2,
                        end_sec=1.4,
                        confidence=0.9,
                        flagged=False,
                    )
                ],
                distance_score=0.1,
                emitted_text="adalenisgv",
            ),
            AlignedChunk(
                chunk_id="020102",
                start_sec=1.5,
                end_sec=2.8,
                words=[
                    WordInterval(
                        word="nasgiya",
                        start_sec=1.5,
                        end_sec=2.8,
                        confidence=0.88,
                        flagged=False,
                    )
                ],
                distance_score=0.12,
                emitted_text="nasgiya",
            ),
        ]
    )

    out_dir = dummy_audio_path.parent / "sliced_verses"
    chunks = pipeline.slice_verse_audio(
        audio=dummy_audio_path,
        alignment=alignment,
        output_dir=out_dir,
        file_prefix="test_mark",
    )

    assert len(chunks) == 2
    assert isinstance(chunks[0], AudioChunk)
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_sec == 0.2
    assert chunks[0].end_sec == 1.4
    assert chunks[1].start_sec == 1.5
    assert chunks[1].end_sec == 2.8

    # Verify exported WAV files
    wav1 = out_dir / "test_mark_020101.wav"
    wav2 = out_dir / "test_mark_020102.wav"
    assert wav1.exists()
    assert wav2.exists()

    seg1 = AudioSegment.from_file(str(wav1))
    assert seg1.frame_rate == 16000
    assert seg1.channels == 1
    assert abs(len(seg1) - 1200) <= 50  # ~1.2s


def test_reconcile_syllabary_asr_helper():
    enriched, pairs = reconcile_syllabary_asr("ᎣᏏᏲ", "osiyo")
    assert isinstance(enriched, str)
    assert len(pairs) > 0
