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
from transcription.pipelines.dialogue import (
    align_syllabary_ctc,
    align_syllabary_greedy,
)
from transcription.core.audio import AudioChunk
from transcription.core.models.output import ModelOutput


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
        self.device = "cpu"
        self.model = self

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

    def infer(self, audio_input: Any, **kwargs: Any) -> ModelOutput:
        vocab = self.processor.tokenizer.get_vocab()
        # Synthetic logits with greedy token activations for "ohsda nikahlisthiha"
        # 100 frames = 2.0s
        lpz = np.full((100, len(vocab)), -10.0, dtype=np.float32)
        lpz[:, 0] = 0.0  # pad
        # Activate tokens for ohsda: o, h, s, d, a
        tokens_1 = ["o", "h", "s", "d", "a"]
        for idx, tok in enumerate(tokens_1):
            if tok in vocab:
                f = 10 + idx * 3
                lpz[f : f + 2, vocab[tok]] = 8.0
        # Activate tokens for nikahlisthiha: n, i, k, a, h, l, i, s, t, h, i, h, a
        tokens_2 = ["n", "i", "k", "a", "h", "l", "i", "s", "t", "h", "i", "h", "a"]
        for idx, tok in enumerate(tokens_2):
            if tok in vocab:
                f = 40 + idx * 3
                if f + 2 <= 100:
                    lpz[f : f + 2, vocab[tok]] = 8.0

        return ModelOutput(
            lpz=lpz,
            vocab=vocab,
            frame_duration_sec=0.02,
            metadata={"pad_token_id": 0},
        )


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


def test_build_syllabary_word_tier_length_mismatch_raises_value_error():
    from transcription.alignment.models import (
        AlignedChunk,
        AlignmentMetrics,
        AlignmentOutput,
        WordInterval,
    )
    from transcription.pipelines.dialogue import build_syllabary_word_tier

    alignment = AlignmentOutput(
        aligned_chunks=[
            AlignedChunk(
                chunk_id="chunk_001",
                start_sec=0.0,
                end_sec=1.0,
                words=[
                    WordInterval(word="w1", start_sec=0.0, end_sec=0.5),
                    WordInterval(word="w2", start_sec=0.5, end_sec=1.0),
                ],
            )
        ],
        source_id="test",
        raw_tokens=[],
        metrics=AlignmentMetrics(0, 0, 0.0, 0.0, 0, 0, 0),
    )
    syllabary_lookup = {"chunk_001": "ᎣᏏᏲ"}

    with pytest.raises(ValueError, match="Syllabary word index 1 exceeds token bounds"):
        build_syllabary_word_tier(alignment, syllabary_lookup)


def test_build_english_word_tier_length_mismatch_raises_value_error():
    from transcription.alignment.models import (
        AlignedChunk,
        AlignmentMetrics,
        AlignmentOutput,
        WordInterval,
    )
    from transcription.pipelines.dialogue import build_english_word_tier

    alignment = AlignmentOutput(
        aligned_chunks=[
            AlignedChunk(
                chunk_id="chunk_001",
                start_sec=0.0,
                end_sec=1.0,
                words=[
                    WordInterval(word="w1", start_sec=0.0, end_sec=0.5),
                    WordInterval(word="w2", start_sec=0.5, end_sec=1.0),
                ],
            )
        ],
        source_id="test",
        raw_tokens=[],
        metrics=AlignmentMetrics(0, 0, 0.0, 0.0, 0, 0, 0),
    )
    source_lookup = {
        "chunk_001": {"code_switched": {"tokens": [{"english_stem": "Hello"}]}}
    }

    with pytest.raises(
        ValueError, match="Code-switched token index 1 exceeds token bounds"
    ):
        build_english_word_tier(alignment, source_lookup)
