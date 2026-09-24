# -*- coding: utf-8 -*-
"""
test_enrichment_pipeline.py

Unit tests for EnrichmentPipeline in digohwelisgi.pipelines.enrichment.
"""

import json
from unittest.mock import MagicMock

import pytest

from digohwelisgi.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    WordInterval,
)
from digohwelisgi.core.models.output import ModelOutput
from digohwelisgi.pipelines.enrichment import (
    EnrichmentPipeline,
    EnrichmentRecord,
    calculate_cer,
    calculate_relative_improvement,
    enrich_syllabary,
)


def test_calculate_cer():
    # Exact match
    assert calculate_cer("osiyo", "osiyo") == 0.0
    # Empty reference
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("", "abc") == 1.0
    # Substitutions / differences
    cer = calculate_cer("adalenisgv", "athaleniskv")
    assert cer > 0.0


def test_calculate_relative_improvement():
    assert calculate_relative_improvement(0.0, 0.0) == 0.0
    assert calculate_relative_improvement(0.50, 0.25) == 50.0
    assert abs(calculate_relative_improvement(0.20, 0.30) - (-50.0)) < 1e-6


def test_enrich_text_basic():
    pipeline = EnrichmentPipeline()
    # In Cherokee T/TH orthography, Ꮝ has preaspiration hs (atalenihskv)
    record = pipeline.enrich_text(
        syllabary_text="ᎠᏓᎴᏂᏍᎬ",
        emitted_text="atalenihsk",
    )
    assert isinstance(record, EnrichmentRecord)
    assert record.syllabary_text == "ᎠᏓᎴᏂᏍᎬ"
    assert record.base_transliteration == "atalenihskv"
    assert record.emitted_text == "atalenihsk"
    assert record.reconciled_text == "atalenihsk"
    assert len(record.aligned_pairs) == 6


def test_enrich_syllabary_procedural():
    res = enrich_syllabary("Ꮣ Ꭶ", "tha kha")
    assert res == "tha kha"


def test_enrich_audio_with_model_output():
    mock_model_output = MagicMock(spec=ModelOutput)
    mock_model_output.decode_greedy.return_value = "atalenihsk"

    pipeline = EnrichmentPipeline()
    record = pipeline.enrich_audio(
        audio=mock_model_output,
        syllabary_text="ᎠᏓᎴᏂᏍᎬ",
        record_id="rec_1",
    )
    assert record.record_id == "rec_1"
    assert record.emitted_text == "atalenihsk"
    assert record.reconciled_text == "atalenihsk"


def test_reconcile_alignment():
    words = [
        WordInterval(word="a", start_sec=0.0, end_sec=0.5, emitted_word="a"),
        WordInterval(word="ta", start_sec=0.5, end_sec=1.0, emitted_word="tha"),
    ]
    chunk = AlignedChunk(
        chunk_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        words=words,
        distance_score=0.95,
    )
    alignment = AlignmentOutput(
        aligned_chunks=[chunk],
        source_id="test_audio",
        metrics=AlignmentMetrics(
            total_chunks=1,
            matched_chunks=1,
            match_ratio=1.0,
            mean_distance_score=0.95,
            total_ground_truth_chars=3,
            total_emitted_chars=4,
        ),
    )
    syllabary_lookup = {"c1": "Ꭰ Ꮣ"}

    pipeline = EnrichmentPipeline()
    reconciled_words = pipeline.reconcile_alignment(alignment, syllabary_lookup)
    assert len(reconciled_words) == 2
    assert reconciled_words[0].word == "a"
    assert reconciled_words[1].word == "tha"


def test_run_manifest_with_precomputed_emitted(tmp_path):
    records = [
        {
            "id": "item1",
            "syllabary_text": "Ꮣ Ꭶ",
            "emitted_text": "tha kha",
            "target": "tha kha",
            "split": "train",
        },
        {
            "id": "item2",
            "syllabary_text": "ᎠᏓᎴᏂᏍᎬ",
            "emitted_text": "atalenihsk",
            "target": "atalenihsk",
            "split": "test",
        },
    ]

    cache_file = tmp_path / "enrich_manifest.json"
    pipeline = EnrichmentPipeline()
    results, summary = pipeline.run_manifest(
        records=records,
        output_cache_path=cache_file,
        evaluate=True,
    )

    assert len(results) == 2
    assert "train" in summary
    assert "test" in summary
    assert "overall" in summary
    assert summary["overall"]["count"] == 2
    assert summary["overall"]["raw_cer"] == 0.0

    assert cache_file.exists()
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
    assert len(data["records"]) == 2
