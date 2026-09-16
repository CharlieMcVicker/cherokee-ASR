# -*- coding: utf-8 -*-
"""
Unit tests for models.py, distance_metrics.py, and normalizers.py.
"""

import pytest
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.distance_metrics import (
    CharacterErrorRateMetric,
    CustomCallableDistanceMetric,
    DefaultCERDistanceMetric,
    DistanceMetric,
    LevenshteinDistanceMetric,
    calculate_cer,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
    normalize_text_for_alignment,
)


def test_models_instantiation():
    token = TokenEmission(word="osiyo", start_sec=0.5, end_sec=1.2, confidence=0.95)
    assert token.word == "osiyo"
    assert token.start_sec == 0.5
    assert token.end_sec == 1.2
    assert token.confidence == 0.95

    chunk = TextChunk(chunk_id="chunk_01", text="ataleniskv")
    assert chunk.chunk_id == "chunk_01"
    assert chunk.text == "ataleniskv"

    word_interval = WordInterval(
        word="osiyo",
        start_sec=0.5,
        end_sec=1.2,
        confidence=0.95,
        flagged=False,
        emitted_word="osiyo",
    )
    assert word_interval.word == "osiyo"
    assert word_interval.emitted_word == "osiyo"

    aligned_chunk = AlignedChunk(
        chunk_id="chunk_01",
        start_sec=0.5,
        end_sec=1.2,
        words=[word_interval],
        distance_score=0.0,
        emitted_text="osiyo",
    )
    assert aligned_chunk.distance_score == 0.0
    assert len(aligned_chunk.words) == 1

    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.0,
        total_ground_truth_chars=5,
        total_emitted_chars=5,
    )
    assert metrics.match_ratio == 1.0

    output = AlignmentOutput(
        source_id="test_audio.wav",
        aligned_chunks=[aligned_chunk],
        raw_tokens=[token],
        metrics=metrics,
    )
    assert output.source_id == "test_audio.wav"
    assert len(output.aligned_chunks) == 1
    assert len(output.raw_tokens) == 1
    assert output.metrics is not None


def test_ctc_aligner_config_defaults():
    from transcription.alignment.models import CTCAlignerConfig

    cfg = CTCAlignerConfig()
    assert cfg.syncope_tokens == ("a", "e", "i", "o", "u", "v", "t")
    assert cfg.intrusive_tokens == ("h", "'")
    assert cfg.intrusive_max_stride == 4
    assert cfg.enforce_phonotactics is True
    assert cfg.flag_min_confidence == 0.05
    assert cfg.flag_min_char_confidence == 0.0005
    assert cfg.index_duration == 0.02
    assert cfg.min_window_size == 8000
    assert cfg.max_window_size == 100000
    assert cfg.buffer_trail_ms == 300
    assert cfg.buffer_lead_ms == 100
    assert cfg.chunk_seconds == 30.0
    assert cfg.margin_seconds == 1.0
    assert cfg.cache is True
    assert cfg.cache_dir is None


def test_default_cer_distance_metric():
    metric = DefaultCERDistanceMetric()
    assert isinstance(metric, DistanceMetric)

    # Identical strings
    assert metric.compute_cost("osiyo", "osiyo") == 0.0

    # Empty cases
    assert metric.compute_cost("", "") == 0.0
    assert metric.compute_cost("a", "") == 1.0
    assert metric.compute_cost("", "a") == 1.0

    # Non-identical strings
    assert metric.compute_cost("osiyo", "osyo") > 0.0


def test_character_error_rate_metric_and_function():
    metric = CharacterErrorRateMetric()
    assert isinstance(metric, DistanceMetric)

    assert metric.compute_cost("hello", "hello") == 0.0
    assert calculate_cer("hello", "hello") == 0.0

    # 1 edit on length 5 -> 0.2
    assert pytest.approx(calculate_cer("hell", "hello"), 0.01) == 0.2


def test_levenshtein_distance_metric():
    metric = LevenshteinDistanceMetric()
    assert isinstance(metric, DistanceMetric)

    assert metric.compute_cost("osiyo", "osiyo") == 0.0
    assert metric.compute_cost("", "") == 0.0

    # Test with custom substitution weights
    custom_metric = LevenshteinDistanceMetric(substitution_weights={("k", "g"): 0.2})
    cost_kg = custom_metric.compute_cost(hypothesis="ga", reference="ka")
    assert pytest.approx(cost_kg, 0.001) == 0.1

    cost_kp = custom_metric.compute_cost(hypothesis="pa", reference="ka")
    assert pytest.approx(cost_kp, 0.001) == 0.5


def test_custom_callable_distance_metric():
    def my_dist(hyp: str, ref: str) -> float:
        return 0.42

    metric = CustomCallableDistanceMetric(fn=my_dist)
    assert isinstance(metric, DistanceMetric)
    assert metric.compute_cost("a", "b") == 0.42


def test_normalize_text_for_alignment():
    # Hyphens stripped, lowercase, punctuation removed, Cherokee consonants normalized
    res = normalize_text_for_alignment("A-da-le-ni-s-gv.")
    assert "-" not in res
    assert "." not in res
    assert res == "atalenihskv"

    # qu -> gw / kw
    qu_res = normalize_text_for_alignment("quana")
    assert "qu" not in qu_res


def test_normalize_syllabary_and_phonetics():
    assert normalize_syllabary_for_alignment("ho-wa") == "howa"
    assert normalize_syllabary_for_alignment("hi-la") == "hila"

    # Phonetics preserves 'h'
    assert normalize_phonetics_for_alignment("ho-wa") == "howa"
    assert normalize_phonetics_for_alignment("hi-la") == "hila"
