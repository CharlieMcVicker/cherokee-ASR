# -*- coding: utf-8 -*-
"""
test_ctc_aligner.py

Unit tests for CTCSegmentationAligner disk caching, deterministic cache keys,
and get_logits_cached helper.
"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch
import numpy as np
from pydub import AudioSegment
import pytest
import torch

from transcription.alignment.ctc_aligner import (
    CTCSegmentationAligner,
    get_logits_cached,
)


class DummyTokenizer:
    pad_token_id = 0

    def get_vocab(self):
        return {
            "[PAD]": 0,
            "a": 1,
            "e": 2,
            "i": 3,
            "o": 4,
            "u": 5,
            "v": 6,
            "h": 7,
            "'": 8,
        }


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyASRModel:
    def __init__(self, name: str = "dummy_model_v1"):
        self.processor = DummyProcessor()
        self.model_name = name
        self.call_count = 0

    def get_logits(self, samples: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        self.call_count += 1
        num_frames = max(50, len(samples) // 320)
        vocab_size = 9
        # Deterministic logits
        logits = torch.zeros((num_frames, vocab_size), dtype=torch.float32)
        logits[:, 1] = 2.0  # token 'a' has higher logit
        return logits

    def decode(self, lpz: np.ndarray):
        res = MagicMock()
        res.confidence = 0.95
        res.text = "dummy"
        return res


@pytest.fixture
def dummy_audio_file(tmp_path: Path) -> Path:
    audio_path = tmp_path / "test_verse.wav"
    # Create a 2.0-second 16kHz silent audio file
    silence = AudioSegment.silent(duration=2000, frame_rate=16000)
    silence.export(str(audio_path), format="wav")
    return audio_path


def test_get_logits_cached_hit_and_miss(dummy_audio_file: Path, tmp_path: Path):
    cache_dir = tmp_path / "cache"
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model), cache=True, cache_dir=cache_dir
    )

    assert model.call_count == 0

    # 1. First run (miss): forward pass executes and writes to cache
    lpz1, dur1, lead1 = aligner.get_logits_cached(dummy_audio_file)
    assert model.call_count == 1
    assert lpz1.shape[1] == 9
    assert dur1 > 0.0

    # Verify cache file exists
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1

    # 2. Second run (hit): should read from disk and NOT call model
    lpz2, dur2, lead2 = aligner.get_logits_cached(dummy_audio_file)
    assert model.call_count == 1  # No additional model calls!
    np.testing.assert_allclose(lpz1, lpz2, rtol=1e-5, atol=1e-5)
    assert dur1 == dur2
    assert lead1 == lead2


def test_get_logits_cached_disabled(dummy_audio_file: Path, tmp_path: Path):
    cache_dir = tmp_path / "cache_disabled"
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model), cache=False, cache_dir=cache_dir
    )

    # First call
    aligner.get_logits_cached(dummy_audio_file)
    assert model.call_count == 1
    # Cache directory should not contain files
    assert len(list(cache_dir.glob("*.npz"))) == 0

    # Second call without cache: invokes forward pass again
    aligner.get_logits_cached(dummy_audio_file)
    assert model.call_count == 2


def test_standalone_get_logits_cached(dummy_audio_file: Path, tmp_path: Path):
    cache_dir = tmp_path / "cache_standalone"
    model = DummyASRModel()

    lpz1, dur1, lead1 = get_logits_cached(
        audio_input=dummy_audio_file,
        asr_model=cast(Any, model),
        cache=True,
        cache_dir=cache_dir,
    )
    assert model.call_count == 1

    # Cache hit
    lpz2, dur2, lead2 = get_logits_cached(
        audio_input=dummy_audio_file,
        asr_model=cast(Any, model),
        cache=True,
        cache_dir=cache_dir,
    )
    assert model.call_count == 1
    np.testing.assert_allclose(lpz1, lpz2, rtol=1e-5, atol=1e-5)


def test_get_logits_cached_ndarray_and_audiosegment(tmp_path: Path):
    cache_dir = tmp_path / "cache_mem"
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model), cache=True, cache_dir=cache_dir
    )

    # Test with AudioSegment
    seg = AudioSegment.silent(duration=300, frame_rate=16000)
    lpz_seg, _, _ = aligner.get_logits_cached(seg)
    assert model.call_count == 1

    # Hit with same AudioSegment
    lpz_seg_hit, _, _ = aligner.get_logits_cached(seg)
    assert model.call_count == 1
    np.testing.assert_allclose(lpz_seg, lpz_seg_hit, rtol=1e-5, atol=1e-5)

    # Test with np.ndarray
    arr = np.zeros(8000, dtype=np.float32)
    lpz_arr, _, _ = aligner.get_logits_cached(arr)
    assert model.call_count == 2

    # Hit with same np.ndarray
    lpz_arr_hit, _, _ = aligner.get_logits_cached(arr)
    assert model.call_count == 2
    np.testing.assert_allclose(lpz_arr, lpz_arr_hit, rtol=1e-5, atol=1e-5)


def test_cherokee_asr_model_get_logits_sliding_window():
    from transcription.models.asr_model import CherokeeASRModel

    # Mock inner Wav2Vec2 model and processor
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer.get_vocab.return_value = {"[PAD]": 0, "a": 1}

    def fake_processor(speech, sampling_rate=16000):
        res = MagicMock()
        res.input_values = [np.array(speech, dtype=np.float32)]
        return res

    mock_processor.side_effect = fake_processor

    def fake_call(x):
        # x is [1, num_samples]
        n_samples = x.shape[1]
        n_frames = n_samples // 320
        res = MagicMock()
        res.logits = torch.ones((1, n_frames, 10), dtype=torch.float32)
        return res

    mock_model.side_effect = fake_call
    asr = CherokeeASRModel(model=mock_model, processor=mock_processor, device="cpu")

    # Short audio (audio <= chunk_seconds): single pass without chunking
    short_audio = np.zeros(16000 * 5, dtype=np.float32)
    logits_short = asr.get_logits_sliding_window(
        short_audio, chunk_seconds=10.0, margin_seconds=1.0
    )
    assert logits_short.shape[0] == 250
    assert logits_short.shape[1] == 10

    # Long audio (70 seconds with 30s chunk and 1s margin)
    long_audio = np.zeros(16000 * 70, dtype=np.float32)
    logits_long = asr.get_logits_sliding_window(
        long_audio, chunk_seconds=30.0, margin_seconds=1.0
    )
    assert logits_long.shape[1] == 10
    assert logits_long.shape[0] > 0

    # Invalid margin error
    with pytest.raises(ValueError):
        asr.get_logits_sliding_window(long_audio, chunk_seconds=2.0, margin_seconds=1.5)


def test_ctc_aligner_sliding_window_cache_key():
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model), chunk_seconds=30.0, margin_seconds=1.0
    )
    arr = np.zeros(16000, dtype=np.float32)

    key1 = aligner._compute_cache_key(arr, chunk_seconds=30.0, margin_seconds=1.0)
    key2 = aligner._compute_cache_key(arr, chunk_seconds=20.0, margin_seconds=1.0)
    key3 = aligner._compute_cache_key(arr, chunk_seconds=30.0, margin_seconds=2.0)

    assert key1 != key2
    assert key1 != key3


def test_ctc_aligner_full_chapter_verse_and_word_harvesting():
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()

    # Create aligner with mock model
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        syncope_tokens=["a"],
        syncope_penalty=2.0,
        flag_min_confidence=0.1,
    )

    # Audio of 2 seconds
    audio = np.zeros(16000 * 2, dtype=np.float32)

    chunks = [
        TextChunk(chunk_id="verse_1", text="a a"),
        TextChunk(chunk_id="verse_2", text="a"),
    ]

    res = aligner.align(audio, chunks=chunks)
    assert len(res.aligned_chunks) == 2

    # Check verse 1
    v1 = res.aligned_chunks[0]
    assert v1.chunk_id == "verse_1"
    assert v1.start_sec >= 0.0
    assert v1.end_sec >= v1.start_sec
    assert len(v1.words) == 2
    assert v1.words[0].word == "a"
    assert v1.words[1].word == "a"
    assert v1.words[0].start_sec <= v1.words[0].end_sec
    assert isinstance(v1.words[0].confidence, float)
    assert isinstance(v1.words[0].flagged, bool)
    assert v1.emitted_text != ""

    # Check verse 2
    v2 = res.aligned_chunks[1]
    assert v2.chunk_id == "verse_2"
    assert v2.start_sec >= v1.start_sec
    assert len(v2.words) == 1
    assert v2.words[0].word == "a"

    # Check alignment metrics
    assert res.metrics is not None
    assert res.metrics.total_chunks == 2
    assert res.metrics.matched_chunks >= 1
    assert res.metrics.match_ratio > 0.0


def test_ctc_aligner_align_empty_chunks():
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()
    aligner = CTCSegmentationAligner(model=cast(Any, model))
    audio = np.zeros(16000, dtype=np.float32)

    # Empty chunk list
    res = aligner.align(audio, chunks=[])
    assert len(res.aligned_chunks) == 0
    assert res.metrics is not None
    assert res.metrics.total_chunks == 0

    # Chunks with empty text
    chunks = [TextChunk(chunk_id="empty_1", text="")]
    res2 = aligner.align(audio, chunks=chunks)
    assert len(res2.aligned_chunks) == 1
    assert res2.aligned_chunks[0].words == []


def test_ctc_aligner_extract_logits_sliding_window_fallback():
    # Model without get_logits_sliding_window method
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        chunk_seconds=1.0,
        margin_seconds=0.2,
        buffer_lead_ms=0,
        buffer_trail_ms=0,
    )
    long_audio = np.zeros(16000 * 3, dtype=np.float32)

    lpz, dur_sec, lead_offset = aligner.extract_logits(
        long_audio, chunk_seconds=1.0, margin_seconds=0.2, apply_buffers=False
    )
    assert dur_sec == 3.0
    assert lead_offset == 0.0
    assert lpz.ndim == 2
    assert lpz.shape[1] == 9
    assert lpz.shape[0] > 0


def test_ctc_aligner_verse_slice_alignment_and_anomaly_detection(
    dummy_audio_file: Path,
):
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        flag_min_confidence=0.1,
    )

    aligned = aligner.align_verse_slice(
        audio_input=dummy_audio_file,
        chunk_id="020101",
        phonetic_text="a a a",
        syllabary_text="Ꭰ Ꭰ Ꭰ",
    )

    assert aligned.chunk_id == "020101"
    assert aligned.start_sec >= 0.0
    assert aligned.end_sec >= aligned.start_sec
    assert len(aligned.words) == 3
    for w in aligned.words:
        assert w.word == "a"
        assert w.start_sec <= w.end_sec
        assert isinstance(w.confidence, float)
        assert isinstance(w.flagged, bool)
        assert w.emitted_word != ""


def test_ctc_aligner_continuous_chapter_multi_verse_monotonicity():
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        syncope_tokens=["a", "e", "i", "o", "u", "v"],
        syncope_penalty=2.0,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.1,
    )

    audio = np.zeros(16000 * 6, dtype=np.float32)
    chunks = [
        TextChunk(chunk_id="v1", text="a a"),
        TextChunk(chunk_id="v2", text="a a a"),
        TextChunk(chunk_id="v3", text="a"),
    ]

    out = aligner.align(audio, chunks=chunks)
    assert len(out.aligned_chunks) == 3

    prev_end = 0.0
    for idx, c in enumerate(out.aligned_chunks):
        assert c.chunk_id == chunks[idx].chunk_id
        assert c.start_sec >= 0.0
        assert c.end_sec >= c.start_sec
        assert c.start_sec >= prev_end - 0.001
        prev_end = c.end_sec
        assert len(c.words) > 0
        for w in c.words:
            assert w.start_sec <= w.end_sec
            assert w.word == "a"

    assert out.metrics is not None
    assert out.metrics.total_chunks == 3
    assert out.metrics.matched_chunks >= 1
    assert out.metrics.match_ratio > 0.0


def test_ctc_aligner_zero_start_timing_preserved(dummy_audio_file: Path):
    """
    Assert that when ctc_segmentation returns 0.0s for the first token,
    it is not dropped by a strict > 0.0 timing filter.
    """
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        buffer_lead_ms=0,
        buffer_trail_ms=0,
    )

    # Ground truth trellis for 1 word has length 4; word index slice is [1:3]
    fake_timings = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    fake_char_probs = np.full(100, -0.1, dtype=np.float32)
    fake_state_list = ["a"] * 100

    with patch(
        "transcription.alignment.ctc_aligner.ctc_segmentation",
        return_value=(fake_timings, fake_char_probs, fake_state_list),
    ):
        aligned_slice = aligner.align_verse_slice(
            audio_input=dummy_audio_file,
            chunk_id="001",
            phonetic_text="a",
            cache=False,
        )

        assert len(aligned_slice.words) == 1
        w = aligned_slice.words[0]
        assert w.start_sec == 0.0
        assert w.end_sec == 0.02
        assert w.emitted_word == "a"
        assert w.confidence > 0.0


def test_ctc_aligner_unaligned_word_emissions_and_confidence(dummy_audio_file: Path):
    """
    Assert that an unaligned word (empty w_timings / negative timings)
    emits an empty string and 0.0 confidence without reading frame 0.
    """
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        buffer_lead_ms=0,
        buffer_trail_ms=0,
    )

    # Word 1 aligns at 0.1s (trellis slice [1:3]), Word 2 unaligned (slice [3:5])
    fake_timings = np.array([-1.0, 0.1, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    fake_char_probs = np.full(100, -0.05, dtype=np.float32)
    fake_state_list = ["a"] * 100

    with patch(
        "transcription.alignment.ctc_aligner.ctc_segmentation",
        return_value=(fake_timings, fake_char_probs, fake_state_list),
    ):
        aligned_slice = aligner.align_verse_slice(
            audio_input=dummy_audio_file,
            chunk_id="001",
            phonetic_text="a a",
            cache=False,
        )

        assert len(aligned_slice.words) == 2
        w1, w2 = aligned_slice.words
        assert w1.word == "a"
        assert w1.emitted_word == "a"
        assert w1.confidence > 0.0

        # Unaligned word 2
        assert w2.word == "a"
        assert w2.emitted_word == ""
        assert w2.confidence == 0.0
        assert w2.flagged is True
        assert w2.start_sec == w1.end_sec
        assert w2.end_sec == w1.end_sec


def test_ctc_aligner_align_unaligned_word_and_window_size():
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        min_window_size=5000,
        max_window_size=20000,
        buffer_lead_ms=0,
        buffer_trail_ms=0,
    )

    audio = np.zeros(16000 * 2, dtype=np.float32)
    chunks = [TextChunk(chunk_id="v1", text="a a")]

    # Word 1 aligns at 0.0s (slice [1:3]), Word 2 unaligned (slice [3:5])
    fake_timings = np.array([-1.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    fake_char_probs = np.full(100, -0.1, dtype=np.float32)
    fake_state_list = ["a"] * 100
    fake_segments = [(0.0, 0.5, 0.1)]

    with (
        patch(
            "transcription.alignment.ctc_aligner.ctc_segmentation",
            return_value=(fake_timings, fake_char_probs, fake_state_list),
        ) as mock_seg,
        patch(
            "transcription.alignment.ctc_aligner.determine_utterance_segments",
            return_value=fake_segments,
        ),
    ):
        out = aligner.align(audio, chunks=chunks, cache=False)

        # Verify CtcSegmentationParameters received aligner's window sizes
        passed_config = mock_seg.call_args[0][0]
        assert passed_config.min_window_size == 5000
        assert passed_config.max_window_size == 20000

        assert len(out.aligned_chunks) == 1
        c = out.aligned_chunks[0]
        assert len(c.words) == 2
        assert c.words[0].start_sec == 0.0
        assert c.words[0].emitted_word == "a"
        assert c.words[0].confidence > 0.0

        # Unaligned word
        assert c.words[1].emitted_word == ""
        assert c.words[1].confidence == 0.0
        assert c.words[1].flagged is True
