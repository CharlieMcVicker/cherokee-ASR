# -*- coding: utf-8 -*-
"""
test_interview_realignment.py

Tests for code-switched interview alignment and verification of output artifacts
for saving-the-voices/gs_mm.
"""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from digohwelisgi.alignment.models import AlignmentOutput
from digohwelisgi.pipelines.dialogue import align_syllabary_greedy
from digohwelisgi.alignment.tests.test_syllabary_runners import DummyASRModel
from digohwelisgi.core.audio import AudioChunk
from pydub import AudioSegment

SAVING_THE_VOICES_DIR = Path("data/projects/saving-the-voices")
OUTPUT_DIR = SAVING_THE_VOICES_DIR / "output_codeswitched"
BASELINE_DIR = SAVING_THE_VOICES_DIR / "output_greedy"

AUDIT_KEYWORDS = {
    "guy",
    "soldier",
    "jay",
    "charley",
    "mccoy",
    "dry",
    "creek",
    "yeah",
    "ok",
}


@pytest.fixture
def dummy_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "dummy_audio.wav"
    silence = AudioSegment.silent(duration=2000, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


def test_align_syllabary_greedy_codeswitched_unit(dummy_audio: Path, tmp_path: Path):
    """
    Unit test verifying that align_syllabary_greedy with code_switched=True
    builds the English Words tier and properly exports multi-tier Praat TextGrid
    and alignment manifest.
    """
    model = DummyASRModel()
    out_dir = tmp_path / "cs_greedy_out"

    result = align_syllabary_greedy(
        audio=dummy_audio,
        transcript="Guy Soldier: ᎯᏅ JayᎢ ᎣᏏᏍ",
        output_dir=out_dir,
        model=cast(Any, model),
        code_switched=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename="test_cs.TextGrid",
        manifest_filename="test_cs_manifest.json",
    )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1

    # Check TextGrid export
    tg_path = out_dir / "test_cs.TextGrid"
    assert tg_path.exists()
    tg_content = tg_path.read_text(encoding="utf-8")
    assert 'name = "Chunks"' in tg_content
    assert 'name = "Words"' in tg_content
    assert 'name = "Padded Words"' in tg_content
    assert 'name = "Syllabary Words"' in tg_content
    assert 'name = "English Words"' in tg_content
    assert 'name = "Reconciled Words"' in tg_content

    # Check Manifest export
    manifest_path = out_dir / "test_cs_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest["lines"]) == 1
    line = manifest["lines"][0]
    add_tiers = line.get("additional_word_tiers", {})
    assert "English Words" in add_tiers
    assert "Syllabary Words" in add_tiers
    assert "Reconciled Words" in add_tiers


@pytest.mark.skipif(
    not (OUTPUT_DIR / "gs_mm_codeswitched_manifest.json").exists(),
    reason="Artifacts from gs_mm code-switched alignment not found on disk",
)
def test_gs_mm_codeswitched_artifacts_exist_and_valid():
    """
    Integration test asserting that the generated realigned gs_mm artifacts exist,
    have valid structure and metrics, and contain aligned code-switched words.
    """
    tg_path = OUTPUT_DIR / "gs_mm_codeswitched.TextGrid"
    manifest_path = OUTPUT_DIR / "gs_mm_codeswitched_manifest.json"

    assert tg_path.exists(), f"Missing TextGrid at {tg_path}"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    # Verify TextGrid structure
    tg_content = tg_path.read_text(encoding="utf-8")
    assert 'File type = "ooTextFile"' in tg_content
    assert 'Object class = "TextGrid"' in tg_content
    assert 'name = "Chunks"' in tg_content
    assert 'name = "Words"' in tg_content
    assert 'name = "Padded Words"' in tg_content
    assert 'name = "Syllabary Words"' in tg_content
    assert 'name = "English Words"' in tg_content
    assert 'name = "Reconciled Words"' in tg_content
    assert 'name = "Raw ASR Emissions"' in tg_content

    # Verify Manifest structure and metrics
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    metrics = manifest.get("metrics", {})
    assert metrics.get("total_chunks") == 290
    assert metrics.get("matched_chunks") == 290
    assert metrics.get("match_ratio") == 1.0

    lines = manifest.get("lines", [])
    assert len(lines) == 290

    # Verify all key English words are present and have positive duration
    found_keywords = set()
    for line in lines:
        words = line.get("words", [])
        add_tiers = line.get("additional_word_tiers", {})
        eng_tier = add_tiers.get("English Words", [])
        syll_tier = add_tiers.get("Syllabary Words", [])

        for w_idx, w in enumerate(words):
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            dur = end - start

            eng_word = ""
            if w_idx < len(eng_tier):
                eng_word = eng_tier[w_idx].get("word", "")

            syll_word = ""
            if w_idx < len(syll_tier):
                syll_word = syll_tier[w_idx].get("word", "")

            tokens_to_check = set()
            if eng_word:
                tokens_to_check.update(eng_word.lower().split())
            if syll_word:
                tokens_to_check.update(syll_word.lower().split())

            for tok in tokens_to_check:
                cleaned_tok = tok.strip(".,?!:;\"'")
                if cleaned_tok in AUDIT_KEYWORDS:
                    found_keywords.add(cleaned_tok)
                    # Every key word occurrence that matched audio must have valid positive duration
                    if not w.get("flagged", False):
                        assert (
                            dur > 0.0
                        ), f"Zero duration for keyword '{cleaned_tok}' at {start}-{end}"

    assert (
        found_keywords == AUDIT_KEYWORDS
    ), f"Missing keywords: {AUDIT_KEYWORDS - found_keywords}"
