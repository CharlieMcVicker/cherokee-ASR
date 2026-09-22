# -*- coding: utf-8 -*-
"""
test_dialogue_pipeline.py

Unit and integration tests for transcription.pipelines.dialogue.pipeline:
- Ingestion of multi-speaker dialogue text with speaker labels.
- Token discrimination isolating English loanwords from Cherokee phonotactics.
- CTC and Greedy dialogue alignment execution.
- 7-tier Praat TextGrid and JSON manifest exporting (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme).
- Verification of DialogueAlignmentPipeline and convenience procedures.
"""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
from pydub import AudioSegment
import pytest

from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    WordInterval,
)
from transcription.alignment.tests.test_syllabary_runners import DummyASRModel
from transcription.audio.segment import AudioChunk
from transcription.cherokee.codeswitching import CodeSwitchedPreparer, TokenType
from transcription.pipelines.dialogue import (
    DialogueAlignmentPipeline,
    align_dialogue,
    align_syllabary_ctc,
    align_syllabary_greedy,
    build_english_word_tier,
    build_phoneme_tier,
    build_speaker_intervals,
    build_syllabary_word_tier,
    build_turn_intervals,
    export_7tier_textgrid,
)


@pytest.fixture
def dummy_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "dummy_audio.wav"
    silence = AudioSegment.silent(duration=2500, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


def test_dialogue_pipeline_token_discrimination():
    """
    Verifies AC #2: Token discrimination isolates English tokens from Cherokee syncope/intrusion masks.
    """
    preparer = CodeSwitchedPreparer()
    line = "Guy Soldier: ᎯᏅ JayᎢ ᎣᏏᏍ coffee"
    res = preparer.prepare_line(line, strip_speaker=True)

    assert res.speaker == "Guy Soldier"
    tokens = res.tokens
    assert len(tokens) == 4

    # 1. ᎯᏅ -> Cherokee Syllabary
    t0 = tokens[0]
    assert t0.token_type == TokenType.CHEROKEE_SYLLABARY
    assert t0.english_stem is None
    assert any(t0.syncope_mask) or any(t0.intrusion_mask)

    # 2. JayᎢ -> Compound Clitic (English stem 'Jay' + Syllabary 'Ꭲ')
    t1 = tokens[1]
    assert t1.token_type == TokenType.COMPOUND_CLITIC
    assert t1.english_stem == "Jay"
    assert t1.syllabary_clitic == "Ꭲ"
    # English stem receives zero syncope/intrusion
    stem_len = len("tse")
    assert all(not m for m in t1.syncope_mask[:stem_len])
    assert all(not m for m in t1.intrusion_mask[:stem_len])

    # 3. ᎣᏏᏍ -> Cherokee Syllabary
    t2 = tokens[2]
    assert t2.token_type == TokenType.CHEROKEE_SYLLABARY

    # 4. coffee -> English loanword
    t3 = tokens[3]
    assert t3.token_type == TokenType.ENGLISH
    assert t3.english_stem == "coffee"
    assert len(t3.syncope_mask) > 0
    assert all(not m for m in t3.syncope_mask)
    assert all(not m for m in t3.intrusion_mask)


def test_dialogue_pipeline_7tier_export(dummy_audio: Path, tmp_path: Path):
    """
    Verifies AC #1 & AC #3:
    DialogueAlignmentPipeline builds and exports 7-tier Praat TextGrids
    (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme) and JSON manifests.
    """
    model = DummyASRModel()
    out_dir = tmp_path / "dialogue_7tier_out"

    pipeline = DialogueAlignmentPipeline(
        asr_model=cast(Any, model),
        code_switched=True,
        strip_speaker=True,
    )

    transcript = [
        "Guy Soldier: ᎯᏅ JayᎢ ᎣᏏᏍ",
        "Charley McCoy: Ꮭ ᎤᏟ ᏱᎦ",
    ]

    result = pipeline.run(
        audio=dummy_audio,
        transcript=transcript,
        output_dir=out_dir,
        reconcile=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename="test_7tier.TextGrid",
        manifest_filename="test_7tier_manifest.json",
    )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 2

    # Check 7-tier Praat TextGrid
    tg_path = out_dir / "test_7tier.TextGrid"
    assert tg_path.exists()
    tg_content = tg_path.read_text(encoding="utf-8")

    # Assert 7 tiers are present
    assert 'name = "Turn"' in tg_content
    assert 'name = "Speaker"' in tg_content
    assert 'name = "Syllabary"' in tg_content
    assert 'name = "English"' in tg_content
    assert 'name = "Reconciled"' in tg_content
    assert 'name = "CTC Word"' in tg_content
    assert 'name = "Phoneme"' in tg_content

    # Assert legacy compatibility tiers are also preserved
    assert 'name = "Chunks"' in tg_content
    assert 'name = "Words"' in tg_content
    assert 'name = "Padded Words"' in tg_content
    assert 'name = "Syllabary Words"' in tg_content
    assert 'name = "English Words"' in tg_content
    assert 'name = "Reconciled Words"' in tg_content

    # Check content of Speaker and Turn intervals
    assert "Guy Soldier" in tg_content
    assert "Charley McCoy" in tg_content

    # Check JSON manifest export
    manifest_path = out_dir / "test_7tier_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest["lines"]) == 2
    l0 = manifest["lines"][0]
    add_tiers = l0.get("additional_word_tiers", {})
    assert "Syllabary Words" in add_tiers
    assert "English Words" in add_tiers
    assert "Reconciled Words" in add_tiers
    assert "Phoneme" in add_tiers


def test_align_dialogue_convenience_runner(dummy_audio: Path, tmp_path: Path):
    """
    Tests the align_dialogue convenience function.
    """
    model = DummyASRModel()
    out_dir = tmp_path / "dialogue_convenience_out"

    result = align_dialogue(
        audio=dummy_audio,
        transcript="Guy Soldier: ᎯᏅ JayᎢ",
        output_dir=out_dir,
        model=cast(Any, model),
        reconcile=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename="convenience.TextGrid",
        manifest_filename="convenience.json",
    )

    assert isinstance(result, AlignmentOutput)
    assert (out_dir / "convenience.TextGrid").exists()
    assert (out_dir / "convenience.json").exists()


def test_align_syllabary_greedy_dialogue_pipeline(dummy_audio: Path, tmp_path: Path):
    """
    Verifies that align_syllabary_greedy exported from dialogue pipeline works identically
    with DTW fallback and 7-tier / code-switched support.
    """
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
            transcript="Guy Soldier: ᎯᏅ JayᎢ ᎣᏏᏍ",
            output_dir=out_dir,
            model=cast(Any, model),
            code_switched=True,
            export_praat=True,
            export_manifest=True,
            textgrid_filename="greedy_7tier.TextGrid",
            manifest_filename="greedy_manifest.json",
        )

    assert isinstance(result, AlignmentOutput)
    tg_path = out_dir / "greedy_7tier.TextGrid"
    assert tg_path.exists()
    content = tg_path.read_text(encoding="utf-8")
    assert 'name = "Turn"' in content
    assert 'name = "Speaker"' in content
    assert 'name = "Syllabary"' in content
    assert 'name = "English"' in content
    assert 'name = "Reconciled"' in content
    assert 'name = "Phoneme"' in content


def test_build_phoneme_tier_structure():
    """
    Verifies sub-word phoneme breakdown preserves word temporal bounds.
    """
    word = WordInterval(
        word="hatvki",
        start_sec=1.0,
        end_sec=2.0,
        emitted_word="hatvki",
    )
    phones = build_phoneme_tier([word])
    assert len(phones) == 6  # h, a, t, v, k, i
    assert phones[0].start_sec == 1.0
    assert phones[-1].end_sec == 2.0
    symbols = [p.word for p in phones]
    assert symbols == ["h", "a", "t", "v", "k", "i"]
