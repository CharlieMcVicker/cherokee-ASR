"""
Unit tests for NeedlemanWunschWordAligner and SlidingWindowDTWAligner.
"""

import pytest
from transcription.alignment.core.word_aligner import NeedlemanWunschWordAligner
from transcription.alignment.core.sliding_window import SlidingWindowDTWAligner
from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.ports.protocols import ChunkAlignmentEngine
from transcription.alignment.strategies.distance_metrics import (
    DefaultCERDistanceMetric,
    PhonologicalDistanceMetric,
)
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
)


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
    raw_syll = ["Ꭳ", "Ꮟ", "Ᏺ"]

    intervals = aligner.align_words(raw_words, tokens, raw_syllabary_words=raw_syll)
    assert len(intervals) == 2
    assert intervals[0].word == "o si"
    assert intervals[0].syllabary == "Ꭳ Ꮟ"
    assert intervals[0].start_sec == 0.0
    assert intervals[0].end_sec == 0.5

    assert intervals[1].word == "yo"
    assert intervals[1].syllabary == "Ᏺ"
    assert intervals[1].start_sec == 0.6
    assert intervals[1].end_sec == 1.0


def test_word_aligner_with_custom_metric_and_preprocessor():
    custom_metric = PhonologicalDistanceMetric()
    custom_prep = CherokeePhoneticPreprocessor()
    aligner = NeedlemanWunschWordAligner(
        distance_metric=custom_metric,
        preprocessor=custom_prep,
    )
    raw_words = ["quana"]
    tokens = [TokenEmission(word="gwana", start_sec=0.0, end_sec=0.5, confidence=0.8)]
    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 1
    assert intervals[0].word == "quana"
    assert intervals[0].start_sec == 0.0


def test_sliding_window_dtw_aligner_protocol():
    aligner = SlidingWindowDTWAligner()
    assert isinstance(aligner, ChunkAlignmentEngine)


def test_sliding_window_dtw_alignment_flow():
    aligner = SlidingWindowDTWAligner()

    chunks = [
        TextChunk(
            chunk_id="chunk_01",
            raw_text="A-da-le-ni-s-gv",
            syllabary_text="ᎠᏓᎴᏂᏍᎬ",
            metadata={"chapter": 1, "verse": 1},
        ),
        TextChunk(
            chunk_id="chunk_02",
            raw_text="yi-s-dv",
            syllabary_text="ᏱᏍᏛ",
            metadata={"chapter": 1, "verse": 2},
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

    out = aligner.align_chunks(emissions=emissions, chunks=chunks)
    assert isinstance(out, AlignmentOutput)
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
    chunks = [TextChunk(chunk_id="c1", raw_text="osiyo")]

    # Empty emissions
    out = aligner.align_chunks(emissions=[], chunks=chunks)
    assert len(out.aligned_chunks) == 1
    assert out.aligned_chunks[0].start_sec == 0.0
    assert out.metrics is not None
    assert out.metrics.matched_chunks == 0
    assert out.metrics.match_ratio == 0.0

    # Empty chunks
    out2 = aligner.align_chunks(
        emissions=[TokenEmission(word="osiyo", start_sec=0.0, end_sec=1.0)],
        chunks=[],
    )
    assert len(out2.aligned_chunks) == 0
    assert out2.metrics is not None
    assert out2.metrics.total_chunks == 0


def test_sliding_window_reconciliation_mock():
    class DummyReconciliation:
        def reconcile(self, syllabary_text: str, emitted_text: str):
            return f"reconciled_{emitted_text}", [(syllabary_text, emitted_text)]

    aligner = SlidingWindowDTWAligner(reconciliation_strategy=DummyReconciliation())
    chunks = [
        TextChunk(
            chunk_id="c1",
            raw_text="osiyo",
            syllabary_text="ᎣᏏᏲ",
        )
    ]
    emissions = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.6, confidence=0.9)
    ]
    out = aligner.align_chunks(
        emissions=emissions,
        chunks=chunks,
    )
    assert len(out.aligned_chunks) == 1
    assert len(out.aligned_chunks[0].words) == 1
    assert out.aligned_chunks[0].words[0].reconciled_word == "reconciled_osiyo"
