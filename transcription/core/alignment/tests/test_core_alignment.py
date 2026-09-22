# -*- coding: utf-8 -*-
"""
test_core_alignment.py

Unit tests for transcription.core.alignment:
- Pure domain models (TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentOutput, CTCAlignerConfig)
- DistanceMetric protocol, DefaultCERDistanceMetric, PhonologicalDistanceMetric, ConfusionMatrixCostMetric
- NeedlemanWunschWordAligner & SlidingWindowDTWAligner consuming ModelOutput & TokenEmission
- CTCSegmentationAligner consuming ModelOutput.lpz with TextPreparerProtocol
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import pytest

from transcription.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.core.alignment.distance import (
    CharacterErrorRateMetric,
    ConfusionMatrixCostMetric,
    CustomCallableDistanceMetric,
    DefaultCERDistanceMetric,
    DistanceMetric,
    PhonologicalDistanceMetric,
    calculate_cer,
)
from transcription.core.alignment.dp import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.core.alignment.ctc import (
    CTCSegmentationAligner,
    TextPreparerProtocol,
    default_text_preparer,
)
from transcription.core.models.output import ModelOutput


def test_models_instantiation_and_properties():
    cfg = CTCAlignerConfig()
    assert cfg.index_duration == 0.02
    assert "t" in cfg.syncope_tokens

    emission = TokenEmission(word="osiyo", start_sec=0.5, end_sec=1.2, confidence=0.98)
    assert emission.word == "osiyo"
    assert emission.start_sec == 0.5
    assert emission.end_sec == 1.2
    assert emission.confidence == 0.98

    chunk = TextChunk(chunk_id="chunk_1", text="osiyo tohiju")
    assert chunk.chunk_id == "chunk_1"
    assert chunk.text == "osiyo tohiju"

    w1 = WordInterval(
        word="osiyo", start_sec=0.5, end_sec=1.0, confidence=0.95, flagged=False
    )
    w2 = WordInterval(
        word="tohiju", start_sec=1.1, end_sec=1.8, confidence=0.40, flagged=True
    )

    aligned = AlignedChunk(
        chunk_id="chunk_1",
        start_sec=0.5,
        end_sec=1.8,
        words=[w1, w2],
        distance_score=0.12,
        emitted_text="osiyo tohiju",
    )
    assert aligned.has_anomalies is True
    assert aligned.flagged_words == [w2]

    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.12,
        total_ground_truth_chars=12,
        total_emitted_chars=12,
        flagged_words_count=1,
    )
    output = AlignmentOutput(
        aligned_chunks=[aligned],
        source_id="test_audio",
        raw_tokens=[emission],
        metrics=metrics,
    )
    assert output.source_id == "test_audio"
    assert len(output.aligned_chunks) == 1
    assert output.metrics is not None
    assert output.metrics.matched_chunks == 1


def test_distance_metrics():
    # CER
    cer_metric = DefaultCERDistanceMetric()
    assert isinstance(cer_metric, DistanceMetric)
    assert cer_metric.compute_cost("osiyo", "osiyo") == 0.0
    assert cer_metric.compute_cost("osiyo", "") == 1.0
    assert cer_metric.compute_cost("", "osiyo") == 1.0
    assert cer_metric.compute_cost("", "") == 0.0
    assert calculate_cer("abc", "abd") > 0.0

    # Phonological
    ph_metric = PhonologicalDistanceMetric(
        substitution_weights={("t", "th"): 0.1, ("k", "kh"): 0.1},
        insertion_cost=0.5,
        deletion_cost=0.5,
        default_substitution_cost=1.0,
    )
    cost_similar = ph_metric.compute_cost("ati", "athi")
    cost_different = ph_metric.compute_cost("ati", "ami")
    assert cost_similar < cost_different

    # Custom callable
    custom = CustomCallableDistanceMetric(lambda h, r: 0.42)
    assert custom.compute_cost("foo", "bar") == 0.42

    # Confusion Matrix
    cm_metric = ConfusionMatrixCostMetric(
        unigram_costs={"a": {"e": 0.2}},
        insertion_cost=1.0,
        deletion_cost=1.0,
        default_substitution_cost=1.0,
    )
    assert cm_metric.compute_cost("e", "a") == 0.2
    assert cm_metric.compute_cost("a", "a") == 0.0


def test_dp_word_aligner():
    aligner = NeedlemanWunschWordAligner(gap_cost=0.8)
    raw_words = ["osiyo", "tohiju"]
    tokens = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.5, confidence=0.9),
        TokenEmission(word="tohiju", start_sec=0.6, end_sec=1.1, confidence=0.85),
    ]
    intervals = aligner.align_words(raw_words, tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "osiyo"
    assert intervals[0].start_sec == 0.1
    assert intervals[0].end_sec == 0.5
    assert intervals[1].word == "tohiju"
    assert intervals[1].start_sec == 0.6
    assert intervals[1].end_sec == 1.1


def test_sliding_window_dtw_aligner_with_token_emissions():
    dtw = SlidingWindowDTWAligner()
    tokens = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.5, confidence=0.95),
        TokenEmission(word="tohiju", start_sec=0.6, end_sec=1.2, confidence=0.90),
    ]
    chunks = [
        TextChunk(chunk_id="chunk_1", text="osiyo"),
        TextChunk(chunk_id="chunk_2", text="tohiju"),
    ]
    output = dtw.align(tokens, chunks, source_id="rec_1")
    assert len(output.aligned_chunks) == 2
    assert output.aligned_chunks[0].chunk_id == "chunk_1"
    assert output.aligned_chunks[0].start_sec == 0.1
    assert output.aligned_chunks[0].end_sec == 0.5
    assert output.aligned_chunks[1].chunk_id == "chunk_2"
    assert output.aligned_chunks[1].start_sec == 0.6
    assert output.aligned_chunks[1].end_sec == 1.2
    assert output.metrics is not None
    assert output.metrics.matched_chunks == 2


def test_sliding_window_dtw_aligner_with_model_output():
    # Build ModelOutput with vocab
    vocab = {"[PAD]": 0, "|": 1, "a": 2, "t": 3, "i": 4}
    # 20 frames, 0.02s per frame = 0.4s
    # Token pattern: | a t i |
    lpz = np.full((20, 5), -10.0, dtype=np.float32)
    lpz[:, 0] = 0.0  # default pad
    lpz[2:4, 2] = 5.0  # 'a'
    lpz[6:8, 3] = 5.0  # 't'
    lpz[10:12, 4] = 5.0  # 'i'

    model_out = ModelOutput(
        lpz=lpz,
        vocab=vocab,
        frame_duration_sec=0.02,
        metadata={"pad_token_id": 0, "word_delimiter_token_id": 1},
    )

    dtw = SlidingWindowDTWAligner()
    chunks = [TextChunk(chunk_id="c1", text="ati")]
    out = dtw.align(model_out, chunks, source_id="utt_1")
    assert len(out.aligned_chunks) == 1
    assert out.aligned_chunks[0].chunk_id == "c1"
    assert len(out.raw_tokens) == 1
    assert out.raw_tokens[0].word == "ati"
    assert out.metrics is not None
    assert out.metrics.matched_chunks == 1


def test_ctc_segmentation_aligner_with_model_output():
    # Mock vocabulary
    char_list = ["[PAD]", "a", "t", "i", "s", "o"]
    pad_id = 0
    vocab = {ch: idx for idx, ch in enumerate(char_list)}

    # Construct synthetic trellis log-probs (50 frames x 6 tokens)
    # Activations for word "ati"
    lpz = np.full((50, len(char_list)), -6.0, dtype=np.float32)
    lpz[:, pad_id] = 0.0  # blank baseline

    # Frame 10: 'a', Frame 20: 't', Frame 30: 'i'
    lpz[8:15, 1] = 6.0
    lpz[18:25, 2] = 6.0
    lpz[28:35, 3] = 6.0

    model_out = ModelOutput(
        lpz=lpz,
        vocab=vocab,
        frame_duration_sec=0.02,
        metadata={"pad_token_id": 0},
    )

    ctc_aligner = CTCSegmentationAligner(
        config=CTCAlignerConfig(
            index_duration=0.02, min_window_size=10, max_window_size=100
        ),
        text_preparer=default_text_preparer,
    )

    chunks = [TextChunk(chunk_id="chunk_ati", text="ati")]
    result = ctc_aligner.align(model_out, chunks, source_id="audio_sample")

    assert len(result.aligned_chunks) == 1
    ac = result.aligned_chunks[0]
    assert ac.chunk_id == "chunk_ati"
    assert ac.start_sec >= 0.0
    assert ac.end_sec > ac.start_sec
    assert len(ac.words) == 1
    assert ac.words[0].word == "ati"
    assert result.metrics is not None
    assert result.metrics.matched_chunks == 1


def test_ctc_segmentation_aligner_custom_preparer():
    call_log = []

    def custom_preparer(
        config: Any,
        text: Union[str, Sequence[str]],
        char_list: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, List[int]]:
        call_log.append((text, kwargs))
        return default_text_preparer(config, text, char_list, **kwargs)

    char_list = ["[PAD]", "a", "b"]
    lpz = np.zeros((20, 3), dtype=np.float32)
    lpz[:, 0] = 2.0
    lpz[5:10, 1] = 5.0

    ctc_aligner = CTCSegmentationAligner(
        config=CTCAlignerConfig(
            index_duration=0.02, min_window_size=5, max_window_size=50
        ),
        text_preparer=custom_preparer,
        char_list=char_list,
        pad_id=0,
    )

    chunks = [TextChunk(chunk_id="c1", text="a")]
    res = ctc_aligner.align(lpz, chunks)
    assert len(call_log) == 1
    assert call_log[0][0] == ["a"]
    assert len(res.aligned_chunks) == 1
