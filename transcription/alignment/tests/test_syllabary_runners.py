# -*- coding: utf-8 -*-
"""
test_syllabary_runners.py

Unit tests for high-level interview alignment runners:
- load_syllabary_transcript
- align_syllabary_greedy
- align_syllabary_ctc
"""

import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
from pydub import AudioSegment
import pytest
import torch

from transcription.alignment.ingestion import load_syllabary_transcript
from transcription.alignment.models import AlignmentOutput, CTCAlignerConfig
from transcription.alignment.pipeline import (
    align_syllabary_ctc,
    align_syllabary_greedy,
)
from transcription.audio.segment import AudioChunk


class DummyTokenizer:
    pad_token_id = 0

    def get_vocab(self):
        chars = [
            "[PAD]",
            "a",
            "e",
            "i",
            "o",
            "u",
            "v",
            "h",
            "'",
            "y",
            "s",
            "t",
            "k",
            "d",
            "g",
            "l",
            "m",
            "n",
            "w",
            "q",
            "j",
        ]
        return {ch: i for i, ch in enumerate(chars)}


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyASRModel:
    def __init__(self, name: str = "dummy_model"):
        self.processor = DummyProcessor()
        self.model_name = name
        self.call_count = 0

    def get_logits(self, samples: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        self.call_count += 1
        num_frames = max(60, len(samples) // 320)
        vocab_size = len(self.processor.tokenizer.get_vocab())
        logits = torch.full((num_frames, vocab_size), -5.0, dtype=torch.float32)
        logits[:, 0] = 0.0  # [PAD]
        for frame in range(10, num_frames, 15):
            logits[frame, 1] = 5.0  # 'a' token
        return logits

    def decode(self, lpz: np.ndarray):
        res = MagicMock()
        res.confidence = 0.95
        res.text = "ohsda"
        return res

    def get_word_confidences(self, audio: Any, sample_rate: int = 16000):
        return [
            MagicMock(word="ohsda", start_time=0.1, end_time=0.6, confidence=0.95),
            MagicMock(
                word="nikahlisthiha", start_time=0.7, end_time=1.4, confidence=0.92
            ),
        ]


@pytest.fixture
def dummy_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "dummy_audio.wav"
    silence = AudioSegment.silent(duration=2000, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


def test_load_syllabary_transcript_string():
    raw_text = "ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ\nᎭᏩ ᎰᏩ"
    chunks, source_lookup = load_syllabary_transcript(raw_text)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "chunk_001"
    assert chunks[1].chunk_id == "chunk_002"
    assert "syllabary" in source_lookup["chunk_001"]
    assert source_lookup["chunk_001"]["syllabary"] == "ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ"


def test_load_syllabary_transcript_list_of_strings():
    lines = ["ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ", "so we were talking about ᎣᏍᏓ"]
    chunks, source_lookup = load_syllabary_transcript(lines)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "chunk_001"
    assert chunks[1].chunk_id == "chunk_002"
    assert "talking" in chunks[1].text
    assert "ohsta" in chunks[1].text


def test_load_syllabary_transcript_json_file(tmp_path: Path):
    json_path = tmp_path / "transcript.json"
    json_path.write_text(
        json.dumps(
            [
                {"id": "c1", "syllabary": "ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ"},
                {"id": "c2", "syllabary": "ᎭᏩ ᎰᏩ"},
            ]
        ),
        encoding="utf-8",
    )
    chunks, source_lookup = load_syllabary_transcript(json_path)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "c1"
    assert chunks[1].chunk_id == "c2"
    assert source_lookup["c1"]["syllabary"] == "ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ"


def test_load_interview_transcript_gs_mm():
    from transcription.alignment.ingestion import load_interview_transcript

    file_path = Path("saving-the-voices/gs_mm.txt")
    if file_path.exists():
        chunks, source_lookup = load_interview_transcript(file_path)
        assert len(chunks) > 100
        first_chunk = chunks[0]
        assert first_chunk.chunk_id == "turn_001"
        assert source_lookup["turn_001"]["speaker"] == "Guy Soldier"
        assert "ᎣᏏᏍ" in source_lookup["turn_001"]["syllabary"]
        # Ensure speaker label was stripped from phonetic alignment tokens
        assert "Guy" not in first_chunk.text


def test_align_syllabary_greedy(dummy_audio: Path, tmp_path: Path):
    model = DummyASRModel()
    out_dir = tmp_path / "greedy_out"

    with patch("transcription.alignment.extractors.segment_long_audio") as mock_segment:
        mock_chunk = AudioChunk(
            chunk_index=0,
            audio=AudioSegment.silent(duration=2000, frame_rate=16000),
            start_sec=0.0,
            end_sec=2.0,
        )
        mock_segment.return_value = [mock_chunk]

        result = align_syllabary_greedy(
            audio=dummy_audio,
            transcript="ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ",
            output_dir=out_dir,
            model=cast(Any, model),
            export_praat=True,
            export_manifest=True,
        )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1

    # Check TextGrid export
    tg_path = out_dir / "greedy_alignment.TextGrid"
    assert tg_path.exists()
    tg_content = tg_path.read_text(encoding="utf-8")
    assert 'name = "Syllabary Words"' in tg_content
    assert 'name = "Reconciled Words"' in tg_content

    # Check Manifest export
    manifest_path = out_dir / "alignment_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["lines"]) == 1
    assert "reconciled_words" in manifest


def test_align_syllabary_ctc(dummy_audio: Path, tmp_path: Path):
    model = DummyASRModel()
    out_dir = tmp_path / "ctc_out"

    result = align_syllabary_ctc(
        audio=dummy_audio,
        transcript="ᎣᏍᏓ ᏂᎦᎵᏍᏗᎭ",
        output_dir=out_dir,
        model=cast(Any, model),
        config=CTCAlignerConfig(cache=False),
        cache=False,
        export_praat=True,
        export_manifest=True,
    )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1

    # Check TextGrid export
    tg_path = out_dir / "ctc_alignment.TextGrid"
    assert tg_path.exists()
    tg_content = tg_path.read_text(encoding="utf-8")
    assert 'name = "Syllabary Words"' in tg_content

    # Check Manifest export
    manifest_path = out_dir / "alignment_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["lines"]) == 1


def test_align_syllabary_runners_code_switching(dummy_audio: Path, tmp_path: Path):
    model = DummyASRModel()
    out_dir = tmp_path / "code_switch_out"

    # Transcript with mixed Cherokee Syllabary and English words
    mixed_transcript = "ᎣᏍᏓ and then we went to the store ᎭᏩ"

    result_ctc = align_syllabary_ctc(
        audio=dummy_audio,
        transcript=mixed_transcript,
        output_dir=out_dir,
        model=cast(Any, model),
        config=CTCAlignerConfig(cache=False),
        cache=False,
        export_praat=True,
        export_manifest=True,
    )

    assert isinstance(result_ctc, AlignmentOutput)
    assert len(result_ctc.aligned_chunks) == 1
    tg_path = out_dir / "ctc_alignment.TextGrid"
    assert tg_path.exists()
