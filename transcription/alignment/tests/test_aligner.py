# -*- coding: utf-8 -*-
"""
Unit tests for aligner.py (NeedlemanWunschWordAligner, SlidingWindowDTWAligner).
"""

import pytest
from transcription.alignment.aligner import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import normalize_phonetics_for_alignment


def test_word_aligner_default_normalizers_identity():
    aligner = NeedlemanWunschWordAligner()
    assert aligner.chunk_norm("HELLO") == "HELLO"
    assert aligner.emission_norm("WORLD") == "WORLD"
    assert aligner.compute_cost("osiyo", "osiyo") == 0.0


def test_word_aligner_custom_normalizers():
    chunk_norm = lambda s: s.lower().replace("-", "")
    emission_norm = lambda s: s.lower().strip()

    aligner = NeedlemanWunschWordAligner(
        chunk_normalizer=chunk_norm,
        emission_normalizer=emission_norm,
    )

    cost = aligner.compute_cost(hypothesis="  OSIYO  ", reference="o-si-yo")
    assert cost == 0.0

    raw_words = ["o-si-yo", "to-hi-ju"]
    tokens = [
        TokenEmission(word="OSIYO", start_sec=0.1, end_sec=0.6, confidence=0.9),
        TokenEmission(word="TOHIJU", start_sec=0.7, end_sec=1.2, confidence=0.85),
    ]
    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "o-si-yo"
    assert intervals[0].start_sec == 0.1
    assert intervals[0].end_sec == 0.6
    assert intervals[1].word == "to-hi-ju"


def test_word_aligner_exact_match():
    aligner = NeedlemanWunschWordAligner()
    raw_words = ["osiyo", "tohiju"]
    tokens = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.6, confidence=0.9),
        TokenEmission(word="tohiju", start_sec=0.7, end_sec=1.2, confidence=0.85),
    ]

    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "osiyo"
    assert intervals[0].start_sec == 0.1
    assert intervals[0].end_sec == 0.6
    assert intervals[0].confidence == 0.9
    assert intervals[1].word == "tohiju"
    assert intervals[1].start_sec == 0.7
    assert intervals[1].end_sec == 1.2


def test_word_aligner_fusion_gt_and_asr():
    aligner = NeedlemanWunschWordAligner()
    # 1 ASR token corresponds to 2 GT words: "o si" -> "osi"
    raw_words = ["o", "si", "yo"]
    tokens = [
        TokenEmission(word="osi", start_sec=0.0, end_sec=0.5, confidence=0.9),
        TokenEmission(word="yo", start_sec=0.6, end_sec=1.0, confidence=0.95),
    ]

    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "o si"
    assert intervals[0].start_sec == 0.0
    assert intervals[0].end_sec == 0.5

    assert intervals[1].word == "yo"
    assert intervals[1].start_sec == 0.6
    assert intervals[1].end_sec == 1.0


def test_word_aligner_empty_and_gap_handling():
    aligner = NeedlemanWunschWordAligner()
    assert aligner.align_words([], []) == []
    assert aligner.align_words(["osiyo"], []) == []
    assert aligner.align_words([], [TokenEmission("osiyo", 0.0, 1.0)]) == []

    # Unaligned GT word creates an interpolated gap entry
    raw_words = ["osiyo", "unmatched_word", "tohiju"]
    tokens = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.6, confidence=0.9),
        TokenEmission(word="tohiju", start_sec=1.0, end_sec=1.5, confidence=0.9),
    ]
    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 3
    assert intervals[0].word == "osiyo"
    assert intervals[1].word == "unmatched_word"
    assert intervals[1].flagged is True
    assert intervals[2].word == "tohiju"


def test_sliding_window_dtw_alignment_flow():
    word_aligner = NeedlemanWunschWordAligner(
        chunk_normalizer=normalize_phonetics_for_alignment,
        emission_normalizer=normalize_phonetics_for_alignment,
    )
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    chunks = [
        TextChunk(
            chunk_id="chunk_01",
            text="A-da-le-ni-s-gv",
        ),
        TextChunk(
            chunk_id="chunk_02",
            text="yi-s-dv",
        ),
    ]

    emissions = [
        # Preamble noise
        TokenEmission(word="intro", start_sec=0.0, end_sec=0.4, confidence=0.5),
        # Chunk 1 match
        TokenEmission(word="adalenisgv", start_sec=0.5, end_sec=1.2, confidence=0.95),
        # Chunk 2 match
        TokenEmission(word="yisdv", start_sec=1.3, end_sec=1.8, confidence=0.90),
    ]

    out = aligner.align(emissions=emissions, chunks=chunks, source_id="test_source")
    assert isinstance(out, AlignmentOutput)
    assert out.source_id == "test_source"
    assert len(out.aligned_chunks) == 2

    c1 = out.aligned_chunks[0]
    assert c1.chunk_id == "chunk_01"
    assert c1.start_sec == 0.5
    assert c1.end_sec == 1.2
    assert c1.distance_score < 0.2
    assert len(c1.words) >= 1

    c2 = out.aligned_chunks[1]
    assert c2.chunk_id == "chunk_02"
    assert c2.start_sec == 1.3
    assert c2.end_sec == 1.8
    assert c2.distance_score < 0.2
    assert len(c2.words) >= 1

    assert out.metrics is not None
    assert out.metrics.total_chunks == 2
    assert out.metrics.matched_chunks == 2
    assert out.metrics.match_ratio == 1.0


def test_sliding_window_empty_inputs():
    aligner = SlidingWindowDTWAligner()
    chunks = [TextChunk(chunk_id="c1", text="osiyo")]

    # Empty emissions
    out = aligner.align(emissions=[], chunks=chunks)
    assert len(out.aligned_chunks) == 1
    assert out.aligned_chunks[0].start_sec == 0.0
    assert out.metrics is not None
    assert out.metrics.matched_chunks == 0
    assert out.metrics.match_ratio == 0.0

    # Empty chunks
    out2 = aligner.align(
        emissions=[TokenEmission(word="osiyo", start_sec=0.0, end_sec=1.0)],
        chunks=[],
    )
    assert len(out2.aligned_chunks) == 0
    assert out2.metrics is not None
    assert out2.metrics.total_chunks == 0


def test_sliding_window_custom_normalizers():
    chunk_norm = lambda s: s.upper().replace("-", "")
    emission_norm = lambda s: s.upper().strip()

    word_aligner = NeedlemanWunschWordAligner(
        chunk_normalizer=chunk_norm,
        emission_normalizer=emission_norm,
    )
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    chunks = [
        TextChunk(chunk_id="c1", text="o-si-yo"),
    ]
    emissions = [
        TokenEmission(word="osiyo", start_sec=0.5, end_sec=1.0, confidence=0.9),
    ]

    out = aligner.align(emissions=emissions, chunks=chunks)
    assert len(out.aligned_chunks) == 1
    assert out.aligned_chunks[0].chunk_id == "c1"
    assert out.aligned_chunks[0].distance_score == 0.0
    assert out.metrics is not None
    assert out.metrics.matched_chunks == 1
    # total_ground_truth_chars and total_emitted_chars should reflect normalized text
    assert out.metrics.total_ground_truth_chars == len("OSIYO")
    assert out.metrics.total_emitted_chars == len("OSIYO")
