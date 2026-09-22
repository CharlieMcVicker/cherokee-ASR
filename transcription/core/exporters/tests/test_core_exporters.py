# -*- coding: utf-8 -*-
"""
Unit tests for transcription.core.exporters (TextGrid, manifest, and debug JSON).
"""

import json
import os
import tempfile
from pathlib import Path

from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TokenEmission,
    WordInterval,
)
from transcription.core.exporters import (
    IntervalTier,
    TextGridBuilder,
    build_contiguous_intervals,
    build_padded_word_intervals,
    export_debug_json,
    export_manifest,
    export_textgrid,
)


def test_interval_tier_and_builder():
    builder = TextGridBuilder(xmin=0.0, xmax=2.0)
    tier1 = IntervalTier(name="Tier1")
    tier1.add_interval(0.0, 1.0, "hello")
    tier1.add_interval(1.0, 2.0, "world")
    builder.add_tier(tier1)

    tier2 = builder.create_tier("Tier2")
    tier2.add_interval(0.5, 1.5, "overlap")

    assert len(builder.tiers) == 2
    assert builder.tiers[0].name == "Tier1"
    assert builder.tiers[1].name == "Tier2"

    content = builder.to_textgrid_string()
    assert 'File type = "ooTextFile"' in content
    assert 'Object class = "TextGrid"' in content
    assert "xmin = 0" in content
    assert "xmax = 2.000" in content
    assert "size = 2" in content
    assert 'name = "Tier1"' in content
    assert 'name = "Tier2"' in content
    assert 'text = "hello"' in content
    assert 'text = "world"' in content

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test.TextGrid"
        res = builder.write(out_file)
        assert os.path.exists(res)
        with open(res, "r", encoding="utf-8") as f:
            written_content = f.read()
        assert written_content == content


def test_build_contiguous_and_padded_intervals():
    raw = [
        {"start_sec": 0.5, "end_sec": 1.0, "text": "one"},
        {"start_sec": 1.2, "end_sec": 1.8, "text": "two"},
    ]
    contiguous = build_contiguous_intervals(raw, total_end=2.0)
    # Gaps should be filled with empty text
    assert len(contiguous) == 5
    assert contiguous[0] == (0.0, 0.5, "")
    assert contiguous[1] == (0.5, 1.0, "one")
    assert contiguous[2] == (1.0, 1.2, "")
    assert contiguous[3] == (1.2, 1.8, "two")
    assert contiguous[4] == (1.8, 2.0, "")

    padded = build_padded_word_intervals(raw, total_end=2.0, pad_sec=0.1)
    # one: 0.5 - 1.1; gap: 1.1 - 1.2; two: 1.2 - 1.9; gap: 1.9 - 2.0
    assert len(padded) == 5
    assert padded[1] == (0.5, 1.1, "one")


def test_export_textgrid_parity():
    w1 = WordInterval(
        word="word1",
        start_sec=1.0,
        end_sec=1.5,
        emitted_word="word1",
    )
    rec_w1 = WordInterval(
        word="rec_word1",
        start_sec=1.0,
        end_sec=1.5,
    )
    aligned = AlignedChunk(
        chunk_id="chunk_01",
        start_sec=1.0,
        end_sec=1.5,
        words=[w1],
        distance_score=0.05,
    )
    raw_token = TokenEmission(word="word1", start_sec=1.0, end_sec=1.5)
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        raw_tokens=[raw_token],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tg_path = export_textgrid(
            alignment=output,
            output_dir=tmpdir,
            source_metadata={"chunk_01": {"text": "A-da-le-ni-s-gv"}},
            additional_word_tiers={"Reconciled Words": [rec_w1]},
        )
        assert os.path.exists(tg_path)

        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert 'File type = "ooTextFile"' in content
        assert 'name = "Chunks"' in content
        assert 'name = "Words"' in content
        assert 'name = "Padded Words"' in content
        assert 'name = "Reconciled Words"' in content
        assert 'name = "Raw ASR Emissions"' in content
        assert "size = 5" in content


def test_export_manifest():
    w1 = WordInterval(
        word="adalenisgv",
        start_sec=1.0,
        end_sec=1.5,
        confidence=0.98,
        flagged=False,
        emitted_word="adalenisgv_emit",
    )
    aligned = AlignedChunk(
        chunk_id="chunk_01",
        start_sec=1.0,
        end_sec=1.5,
        words=[w1],
        distance_score=0.0,
        emitted_text="adalenisgv",
    )
    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.0,
        total_ground_truth_chars=10,
        total_emitted_chars=10,
    )
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        metrics=metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = export_manifest(
            alignment=output,
            output_dir=tmpdir,
            source_metadata={
                "chunk_01": {
                    "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ",
                    "text": "A-da-le-ni-s-gv",
                    "english": "The beginning",
                }
            },
        )
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audio_source"] == "test.wav"
        assert data["metrics"]["total_chunks"] == 1
        assert len(data["lines"]) == 1
        assert data["lines"][0]["line_id"] == "chunk_01"
        assert data["lines"][0]["english"] == "The beginning"
        assert data["lines"][0]["cherokee_syllabary"] == "ᎠᏓᎴᏂᏍᎬ"
        assert data["lines"][0]["words"][0]["emitted_word"] == "adalenisgv_emit"


def test_export_manifest_with_additional_word_tiers():
    w1 = WordInterval(
        word="adalenisgv",
        start_sec=1.0,
        end_sec=1.5,
        confidence=0.98,
        flagged=False,
        emitted_word="adalenisgv_emit",
    )
    rec_w1 = WordInterval(
        word="àdalénisgv",
        start_sec=1.0,
        end_sec=1.5,
        confidence=0.98,
        flagged=False,
    )
    aligned = AlignedChunk(
        chunk_id="chunk_01",
        start_sec=1.0,
        end_sec=1.5,
        words=[w1],
        distance_score=0.0,
        emitted_text="adalenisgv",
    )
    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.0,
        total_ground_truth_chars=10,
        total_emitted_chars=10,
    )
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        metrics=metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = export_manifest(
            alignment=output,
            output_dir=tmpdir,
            source_metadata={
                "chunk_01": {
                    "cherokee": "ᎠᏓᎴᏂᏍᎬ",
                    "text": "A-da-le-ni-s-gv",
                    "english": "The beginning",
                }
            },
            additional_word_tiers={"Reconciled Words": [rec_w1]},
        )
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audio_source"] == "test.wav"
        assert "additional_word_tiers" in data
        assert "Reconciled Words" in data["additional_word_tiers"]
        assert "reconciled_words" in data
        assert data["reconciled_words"][0]["word"] == "àdalénisgv"

        line = data["lines"][0]
        assert line["line_id"] == "chunk_01"
        assert line["cherokee_syllabary"] == "ᎠᏓᎴᏂᏍᎬ"
        assert line["words"][0]["reconciled_word"] == "àdalénisgv"
        assert "additional_word_tiers" in line
        assert "reconciled_words" in line
        assert line["reconciled_words"][0]["word"] == "àdalénisgv"


def test_export_debug_json():
    raw_token = TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.5, confidence=0.9)
    aligned = AlignedChunk(
        chunk_id="c1",
        start_sec=0.1,
        end_sec=0.5,
        words=[],
        distance_score=0.0,
        emitted_text="osiyo",
    )
    metrics = AlignmentMetrics(1, 1, 1.0, 0.0, 5, 5)
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        raw_tokens=[raw_token],
        metrics=metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        debug_path = export_debug_json(alignment=output, output_dir=tmpdir)
        assert os.path.exists(debug_path)

        with open(debug_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audio_source"] == "test.wav"
        assert data["aligned_chunks_count"] == 1
        assert len(data["raw_tokens"]) == 1
        assert data["metrics"]["matched_chunks"] == 1
