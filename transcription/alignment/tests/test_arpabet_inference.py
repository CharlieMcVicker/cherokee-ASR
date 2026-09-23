# -*- coding: utf-8 -*-
"""
transcription.alignment.tests.test_arpabet_inference

Comprehensive unit tests for batch Cherokee ASR inference runner, duration-sorted
bucketing, phonotactic digraph grouping with confidence averaging, CTC prefix
beam search Top-3 extraction, and emissions cache persistence.
"""

from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from transcription.cherokee.arpabet.inference import (
    bucket_by_duration,
    ctc_prefix_beam_search,
    decode_greedy_with_confidences,
    extract_top_k_hypotheses,
    group_digraphs_with_confidences,
    run_model_inference_on_manifest,
    sanitize_model_id,
)
from transcription.cherokee.arpabet.types import (
    CherokeeToken,
    InferenceCacheManifest,
    TopKHypothesis,
    WordInferenceCacheEntry,
    WordManifestEntry,
)
from transcription.cherokee.models import CherokeeASRModel
from transcription.cherokee.orthography import Orthography

# ============================================================================
# Dummy / Helper Objects for Testing
# ============================================================================


class DummyTokenizer:
    """Mock Wav2Vec2 tokenizer for unit testing CTC decoders."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.word_delimiter_token_id = 1
        # vocab: 0: [PAD], 1: |, 2: h, 3: s, 4: k, 5: o, 6: w, 7: i, 8: t, 9: a
        self.vocab = {
            "[PAD]": 0,
            "|": 1,
            "h": 2,
            "s": 3,
            "k": 4,
            "o": 5,
            "w": 6,
            "i": 7,
            "t": 8,
            "a": 9,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def decode(self, token_ids: Sequence[int]) -> str:
        res = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if tok == "|":
                res.append(" ")
            elif tok != "[PAD]":
                res.append(tok)
        return "".join(res)


class DummyProcessor:
    """Mock Wav2Vec2Processor wrapper."""

    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids)


def _make_dummy_manifest_entry(
    clip_id: str,
    word: str,
    duration: float,
    audio_path: str = "dummy.wav",
) -> WordManifestEntry:
    """Helper to construct dummy WordManifestEntry instances."""
    return WordManifestEntry(
        clip_id=clip_id,
        audio_path=audio_path,
        word=word,
        duration=duration,
        arpabet=(),
    )


# ============================================================================
# 1. Model ID Sanitization Tests
# ============================================================================


def test_sanitize_model_id_standard():
    assert (
        sanitize_model_id("charliemcvicker/cherokee-wav2vec2")
        == "charliemcvicker_cherokee-wav2vec2"
    )


def test_sanitize_model_id_with_revision():
    assert (
        sanitize_model_id("charliemcvicker/length-only-asr-bible_57a318")
        == "charliemcvicker_length-only-asr-bible_57a318"
    )


def test_sanitize_model_id_with_special_characters():
    assert (
        sanitize_model_id("facebook/wav2vec2-base-960h:v1.0")
        == "facebook_wav2vec2-base-960h_v1.0"
    )
    assert (
        sanitize_model_id("/Users/test/models/cherokee asr:final")
        == "Users_test_models_cherokee_asr_final"
    )


def test_sanitize_model_id_empty_fallback():
    assert sanitize_model_id("") == "default_model"
    assert sanitize_model_id("///") == "default_model"


# ============================================================================
# 2. Duration-Sorted Batching Tests
# ============================================================================


def test_bucket_by_duration_sorting_and_partitioning():
    entries = [
        _make_dummy_manifest_entry("w1", "ONE", 0.95),
        _make_dummy_manifest_entry("w2", "TWO", 0.20),
        _make_dummy_manifest_entry("w3", "THREE", 0.55),
        _make_dummy_manifest_entry("w4", "FOUR", 0.12),
        _make_dummy_manifest_entry("w5", "FIVE", 0.80),
    ]

    batches = bucket_by_duration(entries, batch_size=2)
    assert len(batches) == 3
    # Batch 1: 0.12, 0.20
    assert [e.clip_id for e in batches[0]] == ["w4", "w2"]
    # Batch 2: 0.55, 0.80
    assert [e.clip_id for e in batches[1]] == ["w3", "w5"]
    # Batch 3: 0.95
    assert [e.clip_id for e in batches[2]] == ["w1"]


def test_bucket_by_duration_edge_cases():
    # Empty list
    assert bucket_by_duration([], batch_size=32) == []

    # Single item
    single = [_make_dummy_manifest_entry("w1", "ONE", 0.5)]
    assert bucket_by_duration(single, batch_size=10) == [single]

    # Batch size larger than dataset
    entries = [
        _make_dummy_manifest_entry("w1", "ONE", 0.5),
        _make_dummy_manifest_entry("w2", "TWO", 0.3),
    ]
    batches = bucket_by_duration(entries, batch_size=100)
    assert len(batches) == 1
    assert [e.clip_id for e in batches[0]] == ["w2", "w1"]

    # Invalid batch_size <= 0 raises ValueError
    with pytest.raises(ValueError):
        bucket_by_duration(entries, batch_size=0)
    with pytest.raises(ValueError):
        bucket_by_duration(entries, batch_size=-5)


def test_bucket_by_duration_with_dicts():
    items = [
        {"id": 1, "duration": 1.2},
        {"id": 2, "duration": 0.4},
        {"id": 3, "duration": 0.8},
    ]
    batches = bucket_by_duration(items, batch_size=2)
    assert len(batches) == 2
    assert [it["id"] for it in batches[0]] == [2, 3]
    assert [it["id"] for it in batches[1]] == [1]


# ============================================================================
# 3. Digraph & Trigraph Phonotactic Grouping Tests
# ============================================================================


def test_group_digraphs_hskowi():
    # Target from prompt: "hskowi" -> ["hs", "k", "o", "w", "i"]
    text = "hskowi"
    # Individual character confidences:
    # h: 0.90, s: 0.80 -> hs average = 0.85
    # k: 0.95
    # o: 0.85
    # w: 0.75
    # i: 0.90
    char_confs = [0.90, 0.80, 0.95, 0.85, 0.75, 0.90]

    tokens, confs = group_digraphs_with_confidences(text, char_confs)

    assert len(tokens) == 5
    assert [t.phone for t in tokens] == ["hs", "k", "o", "w", "i"]
    assert all(isinstance(t, CherokeeToken) for t in tokens)
    assert all(t.orthography == Orthography.TTH for t in tokens)

    # Average of h (0.90) and s (0.80) is 0.85
    assert confs[0] == pytest.approx(0.85)
    assert confs[1] == pytest.approx(0.95)
    assert confs[2] == pytest.approx(0.85)
    assert confs[3] == pytest.approx(0.75)
    assert confs[4] == pytest.approx(0.90)


def test_group_all_canonical_digraphs_and_trigraphs():
    # Test all 11 canonical digraphs/trigraphs:
    # hs, th, kh, ts, tsh, tl, tlh, lh, nh, wh, yh
    targets = [
        ("hs", [0.8, 0.9], 0.85),
        ("th", [0.7, 0.9], 0.80),
        ("kh", [0.6, 0.8], 0.70),
        ("ts", [0.9, 0.9], 0.90),
        ("tsh", [0.6, 0.7, 0.8], 0.70),
        ("tl", [0.8, 0.8], 0.80),
        ("tlh", [0.7, 0.8, 0.9], 0.80),
        ("lh", [0.85, 0.95], 0.90),
        ("nh", [0.75, 0.85], 0.80),
        ("wh", [0.9, 1.0], 0.95),
        ("yh", [0.8, 0.8], 0.80),
    ]

    for digraph, raw_confs, expected_avg in targets:
        tokens, confs = group_digraphs_with_confidences(digraph, raw_confs)
        assert len(tokens) == 1, f"Failed for {digraph}"
        assert tokens[0].phone == digraph
        assert confs[0] == pytest.approx(expected_avg)


def test_group_digraphs_empty_and_whitespace():
    # Empty string
    tokens, confs = group_digraphs_with_confidences("", [])
    assert tokens == ()
    assert confs == ()

    # Whitespace handling: "ithahso atil"
    text = "ithahso atil"
    char_confs = [0.9] * len(text)
    tokens, confs = group_digraphs_with_confidences(text, char_confs)
    # Tokens should skip the space:
    # ithahso: i (0:1), th (1:3), a (3:4), hs (4:6), o (6:7) -> 5 tokens
    # atil: a (8:9), t (9:10), i (10:11), l (11:12) -> 4 tokens
    assert [t.phone for t in tokens] == [
        "i",
        "th",
        "a",
        "hs",
        "o",
        "a",
        "t",
        "i",
        "l",
    ]
    assert len(confs) == 9


# ============================================================================
# 4. CTC Prefix Beam Search & Top-K Hypothesis Tests
# ============================================================================


def test_ctc_prefix_beam_search_synthetic():
    processor = DummyProcessor()
    # Construct synthetic logits [T=6, V=10]
    # We want to force greedy path to emit 'h', 's' ("hs")
    # Vocab: 0: PAD, 1: |, 2: h, 3: s, 4: k, 5: o, 6: w, 7: i, 8: t, 9: a
    logits = np.zeros((6, 10), dtype=np.float32)

    # Frame 0: blank (PAD=0)
    logits[0, 0] = 10.0
    # Frame 1: 'h' (id=2)
    logits[1, 2] = 10.0
    # Frame 2: blank (PAD=0)
    logits[2, 0] = 10.0
    # Frame 3: 's' (id=3)
    logits[3, 3] = 10.0
    # Frame 4: 's' continuation (id=3)
    logits[4, 3] = 10.0
    # Frame 5: blank (PAD=0)
    logits[5, 0] = 10.0

    hyps = ctc_prefix_beam_search(logits, processor, beam_width=5, top_k=3)
    assert len(hyps) >= 1
    rank1 = hyps[0]
    assert rank1[0] == 1  # rank
    assert rank1[1] == "hs"  # decoded text
    assert rank1[2] <= 0.0  # log probability score is non-positive


def test_extract_top_k_hypotheses():
    processor = DummyProcessor()
    # Construct synthetic logits [T=8, V=10]
    # Emits 'h', 's', 'k', 'o' ("hsko")
    logits = np.zeros((8, 10), dtype=np.float32)
    # Frame 0: PAD
    logits[0, 0] = 5.0
    # Frame 1: 'h'
    logits[1, 2] = 8.0
    logits[1, 8] = 4.0  # alternative 't'
    # Frame 2: 's'
    logits[2, 3] = 8.0
    # Frame 3: PAD
    logits[3, 0] = 5.0
    # Frame 4: 'k'
    logits[4, 4] = 8.0
    # Frame 5: PAD
    logits[5, 0] = 5.0
    # Frame 6: 'o'
    logits[6, 5] = 8.0
    # Frame 7: PAD
    logits[7, 0] = 5.0

    top_hyps = extract_top_k_hypotheses(logits, processor, beam_width=10, top_k=3)

    assert len(top_hyps) == 3
    # Verify rankings
    assert [h.rank for h in top_hyps] == [1, 2, 3]

    # Verify score monotonicity (descending)
    assert top_hyps[0].score >= top_hyps[1].score >= top_hyps[2].score

    # Rank 1 should be 'hsko' with tokens ['hs', 'k', 'o']
    rank1 = top_hyps[0]
    assert rank1.text == "hsko"
    assert [t.phone for t in rank1.tokens] == ["hs", "k", "o"]
    assert len(rank1.token_confidences) == 3

    # Test roundtrip serialization of TopKHypothesis
    d = rank1.to_dict()
    assert TopKHypothesis.from_dict(d) == rank1


# ============================================================================
# 5. Greedy Decoding with Confidences Tests
# ============================================================================


def test_decode_greedy_with_confidences():
    processor = DummyProcessor()
    # Logits emitting 'k', 'o', 'w', 'i' ("kowi")
    # Vocab: 0: PAD, 4: k, 5: o, 6: w, 7: i
    logits = np.zeros((5, 10), dtype=np.float32)
    logits[0, 4] = 10.0  # k
    logits[1, 5] = 10.0  # o
    logits[2, 6] = 10.0  # w
    logits[3, 7] = 10.0  # i
    logits[4, 0] = 10.0  # PAD

    text, tokens, confs, mean_conf = decode_greedy_with_confidences(logits, processor)

    assert text == "kowi"
    assert [t.phone for t in tokens] == ["k", "o", "w", "i"]
    assert len(confs) == 4
    assert mean_conf >= 0.99


# ============================================================================
# 6. Cache Hit/Miss & Manifest Inference Tests
# ============================================================================


def test_run_model_inference_cache_miss_and_hit(tmp_path: Path):
    processor = DummyProcessor()

    # Create dummy model with mock get_logits_batch
    mock_model = MagicMock(spec=CherokeeASRModel)
    mock_model.processor = processor
    mock_model.model_name = "test_cherokee_model/checkpoint_01"
    mock_model.device = "cpu"

    # Synthetic logits tensor for 1 clip
    # Vocab: 0: PAD, 2: h, 3: s, 4: k, 5: o, 6: w, 7: i ("hskowi")
    synth_logits = torch.zeros((6, 10), dtype=torch.float32)
    synth_logits[0, 2] = 10.0  # h
    synth_logits[1, 3] = 10.0  # s
    synth_logits[2, 4] = 10.0  # k
    synth_logits[3, 5] = 10.0  # o
    synth_logits[4, 6] = 10.0  # w
    synth_logits[5, 7] = 10.0  # i

    mock_model.get_logits_batch.return_value = [synth_logits]

    manifest = [_make_dummy_manifest_entry("clip_001", "WORD", 0.45, "dummy1.wav")]

    cache_dir = tmp_path / "cache"

    # 1. First run: Cache MISS -> forward pass executes and writes cache
    res1 = run_model_inference_on_manifest(
        model=mock_model,
        manifest=manifest,
        batch_size=16,
        cache_dir=cache_dir,
        force_recompute=False,
        show_progress=False,
    )

    assert mock_model.get_logits_batch.call_count == 1
    assert len(res1) == 1
    entry = res1.get("clip_001")
    assert entry is not None
    assert entry.word == "WORD"
    assert entry.greedy_text == "hskowi"
    assert [t.phone for t in entry.greedy_tokens] == ["hs", "k", "o", "w", "i"]
    assert len(entry.top_hypotheses) >= 1

    # Verify cache file was written to disk
    sanitized = sanitize_model_id(mock_model.model_name)
    cache_file = cache_dir / f"{sanitized}_emissions.json"
    assert cache_file.exists()

    # 2. Second run: Cache HIT -> forward pass is completely bypassed
    mock_model.get_logits_batch.reset_mock()
    res2 = run_model_inference_on_manifest(
        model=mock_model,
        manifest=manifest,
        batch_size=16,
        cache_dir=cache_dir,
        force_recompute=False,
        show_progress=False,
    )

    # get_logits_batch should NOT be called on cache hit
    assert mock_model.get_logits_batch.call_count == 0
    assert len(res2) == 1
    cached_entry = res2.get("clip_001")
    assert cached_entry is not None
    assert cached_entry.greedy_text == "hskowi"
    assert cached_entry.greedy_tokens == entry.greedy_tokens

    # 3. Third run: force_recompute=True -> forward pass executes again
    mock_model.get_logits_batch.reset_mock()
    res3 = run_model_inference_on_manifest(
        model=mock_model,
        manifest=manifest,
        batch_size=16,
        cache_dir=cache_dir,
        force_recompute=True,
        show_progress=False,
    )

    assert mock_model.get_logits_batch.call_count == 1
    assert len(res3) == 1
