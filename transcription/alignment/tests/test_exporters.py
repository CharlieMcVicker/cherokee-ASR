# -*- coding: utf-8 -*-
"""
Unit tests for exporters.py.
"""

import json
import os
import tempfile

from transcription.alignment.exporters import (
    export_debug_json,
    export_manifest,
    export_textgrid,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)


def test_export_textgrid():
    w1 = WordInterval(
        word="word1",
        start_sec=1.0,
        end_sec=1.5,
        reconciled_word="rec_word1",
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
        reconciled_word="adalenisgv_rec",
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
        assert data["lines"][0]["words"][0]["reconciled_word"] == "adalenisgv_rec"


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
