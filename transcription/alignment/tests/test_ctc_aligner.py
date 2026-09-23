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
from transcription.alignment.models import CTCAlignerConfig, TextChunk
from transcription.alignment.phonotactics import prepare_cherokee_text


class DummyTokenizer:
    pad_token_id = 0

    def get_vocab(self):
        chars = [
            "[PAD]",
            "a",
            "e",
            "i",
            "o",
            "u",
            "v",
            "h",
            "'",
            "y",
            "s",
            "t",
            "k",
            "d",
            "g",
            "l",
            "m",
            "n",
            "w",
            "q",
            "j",
        ]
        return {ch: i for i, ch in enumerate(chars)}


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
        num_frames = max(60, len(samples) // 320)
        vocab_size = len(self.processor.tokenizer.get_vocab())
        # Deterministic logits: blank baseline with periodic 'a' activations
        logits = torch.full((num_frames, vocab_size), -5.0, dtype=torch.float32)
        logits[:, 0] = 0.0  # [PAD] / blank baseline
        for frame in range(10, num_frames, 15):
            logits[frame, 1] = 5.0  # 'a' token
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
        model=cast(Any, model),
        config=CTCAlignerConfig(cache=True, cache_dir=cache_dir),
    )

    assert model.call_count == 0

    # 1. First run (miss): forward pass executes and writes to cache
    lpz1, dur1, lead1 = aligner.get_logits_cached(dummy_audio_file)
    assert model.call_count == 1
    assert lpz1.shape[1] == len(model.processor.tokenizer.get_vocab())
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
        model=cast(Any, model),
        config=CTCAlignerConfig(cache=False, cache_dir=cache_dir),
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
        model=cast(Any, model),
        config=CTCAlignerConfig(cache=True, cache_dir=cache_dir),
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
    from transcription.cherokee.models import CherokeeASRModel

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
        model=cast(Any, model),
        config=CTCAlignerConfig(chunk_seconds=30.0, margin_seconds=1.0),
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
        config=CTCAlignerConfig(
            syncope_tokens=("a",),
            flag_min_confidence=0.1,
        ),
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
        config=CTCAlignerConfig(
            chunk_seconds=1.0,
            margin_seconds=0.2,
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )
    long_audio = np.zeros(16000 * 3, dtype=np.float32)

    lpz, dur_sec, lead_offset = aligner.extract_logits(
        long_audio, chunk_seconds=1.0, margin_seconds=0.2, apply_buffers=False
    )
    assert dur_sec == 3.0
    assert lead_offset == 0.0
    assert lpz.ndim == 2
    assert lpz.shape[1] == len(model.processor.tokenizer.get_vocab())
    assert lpz.shape[0] > 0


def test_ctc_aligner_verse_slice_alignment_and_anomaly_detection(
    dummy_audio_file: Path,
):
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        config=CTCAlignerConfig(flag_min_confidence=0.1),
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
        config=CTCAlignerConfig(
            syncope_tokens=("a", "e", "i", "o", "u", "v"),
            intrusive_tokens=("h", "'"),
        ),
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
        config=CTCAlignerConfig(
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )

    # Ground truth trellis for 1 word has length 4; word character index is 2
    fake_timings = np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32)
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
        config=CTCAlignerConfig(
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )

    # Word 1 character aligns at 0.1s (trellis index 2), Word 2 unaligned (trellis index 4)
    fake_timings = np.array([-1.0, -1.0, 0.1, -1.0, -1.0, -1.0], dtype=np.float32)
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
        config=CTCAlignerConfig(
            min_window_size=5000,
            max_window_size=20000,
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )

    audio = np.zeros(16000 * 2, dtype=np.float32)
    chunks = [TextChunk(chunk_id="v1", text="a a")]

    # Word 1 character aligns at 0.0s (trellis index 2), Word 2 unaligned (trellis index 4)
    fake_timings = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
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


def test_ctc_aligner_intra_verse_unaligned_word_isolation(dummy_audio_file: Path):
    """
    Assert that unaligned words do not cause subsequent words to accumulate
    preceding word character states or evaluate to 0.0s start timestamps.
    """
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        config=CTCAlignerConfig(
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )

    # Word 1 (a) aligns at 0.10s (frame 5)
    # Word 2 (a) is unaligned (no timing)
    # Word 3 (a) aligns at 0.30s (frame 15)
    # utt_indices: [1, 3, 5, 7]
    fake_timings = np.array(
        [-1.0, -1.0, 0.10, -1.0, -1.0, -1.0, 0.30, -1.0], dtype=np.float32
    )
    fake_char_probs = np.full(100, -0.05, dtype=np.float32)
    fake_state_list = [""] * 100
    fake_state_list[5] = "a"
    fake_state_list[15] = "a"

    with patch(
        "transcription.alignment.ctc_aligner.ctc_segmentation",
        return_value=(fake_timings, fake_char_probs, fake_state_list),
    ):
        aligned_slice = aligner.align_verse_slice(
            audio_input=dummy_audio_file,
            chunk_id="001",
            phonetic_text="a a a",
            cache=False,
        )

        assert len(aligned_slice.words) == 3
        w1, w2, w3 = aligned_slice.words

        # Word 1
        assert w1.start_sec == 0.10
        assert w1.end_sec == 0.12
        assert w1.emitted_word == "a"
        assert w1.flagged is False

        # Word 2 (unaligned)
        assert w2.start_sec == w1.end_sec
        assert w2.end_sec == w1.end_sec
        assert w2.emitted_word == ""
        assert w2.flagged is True

        # Word 3 (isolated from Word 1 & 2)
        assert w3.start_sec == 0.30
        assert w3.end_sec == 0.32
        assert w3.emitted_word == "a"
        assert w3.flagged is False


def test_ctc_aligner_min_char_confidence_flags_mark_1_1_typo(dummy_audio_file: Path):
    """
    Assert that Mark 1:1 'yihstv' typo is flagged due to min_char_confidence < 0.005,
    even when overall geometric/mean confidence is above flag_min_confidence (0.01).
    """
    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        config=CTCAlignerConfig(
            flag_min_confidence=0.01,
            flag_min_char_confidence=0.005,
            buffer_lead_ms=0,
            buffer_trail_ms=0,
        ),
    )

    # Word timings: ground truth trellis length 10, character timings at 0.10, 0.12, 0.14, 0.16
    fake_timings = np.array(
        [-1.0, -1.0, 0.10, 0.12, 0.14, 0.16, -1.0, -1.0, -1.0, -1.0],
        dtype=np.float32,
    )
    fake_state_list = [""] * 100
    fake_char_probs = np.full(100, 0.0, dtype=np.float32)

    # Character states for 'yihstv' across frames 5..8
    # Suppose 3 characters have logprob -0.5 (prob 0.606), but one typo character has logprob -6.0 (prob 0.00247 < 0.005)
    for f, ch in zip([5, 6, 7, 8], ["y", "i", "h", "s"]):
        fake_state_list[f] = ch
        fake_char_probs[f] = -0.5

    fake_char_probs[7] = -6.0  # low char prob: exp(-6.0) ~= 0.00247875

    with patch(
        "transcription.alignment.ctc_aligner.ctc_segmentation",
        return_value=(fake_timings, fake_char_probs, fake_state_list),
    ):
        aligned_slice = aligner.align_verse_slice(
            audio_input=dummy_audio_file,
            chunk_id="020101",
            phonetic_text="yihstv",
            cache=False,
        )

        assert len(aligned_slice.words) == 1
        w = aligned_slice.words[0]
        assert w.word in ("yihstv", "yihsthv")
        assert w.emitted_word == "yihs"
        # Mean logprob: (-0.5*3 + -6.0)/4 = -1.875 -> exp(-1.875) ~= 0.153 > 0.01
        assert w.confidence > 0.01
        # Minimum char confidence is < 0.005
        assert w.min_char_confidence is not None
        assert w.min_char_confidence < 0.005
        assert round(w.min_char_confidence, 4) == round(float(np.exp(-6.0)), 4)
        # Therefore flagged should be True!
        assert w.flagged is True


def test_ctc_aligner_parameters_forwarding(dummy_audio_file: Path):
    """
    Verify that intrusive_tokens, syncope_tokens, intrusive_max_stride,
    and enforce_phonotactics are forwarded to CtcSegmentationParameters and prepare_cherokee_text.
    """
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()

    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        config=CTCAlignerConfig(
            syncope_tokens=("a", "e"),
            intrusive_tokens=("h", "'"),
            intrusive_max_stride=2,
            enforce_phonotactics=True,
            flag_min_char_confidence=0.008,
        ),
    )

    fake_timings = np.array([-1.0, -1.0, 0.10, -1.0], dtype=np.float32)
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
        patch(
            "transcription.alignment.ctc_aligner.prepare_cherokee_text",
            wraps=prepare_cherokee_text,
        ) as mock_prep,
    ):
        # 1. align_verse_slice
        aligner.align_verse_slice(
            audio_input=dummy_audio_file,
            chunk_id="001",
            phonetic_text="adalenisgv",
            cache=False,
        )

        passed_config = mock_seg.call_args[0][0]
        assert passed_config.syncope_tokens == ["a", "e"]
        assert passed_config.intrusive_tokens == ["h", "'"]
        assert passed_config.intrusive_max_stride == 2
        assert mock_prep.call_args[1]["enforce_phonotactics"] is True
        assert hasattr(passed_config, "is_syncope_token")
        assert hasattr(passed_config, "is_intrusive_site")

        # 2. align
        mock_seg.reset_mock()
        mock_prep.reset_mock()
        audio = np.zeros(16000 * 2, dtype=np.float32)
        aligner.align(
            audio, chunks=[TextChunk(chunk_id="v1", text="adalenisgv")], cache=False
        )

        passed_config2 = mock_seg.call_args[0][0]
        assert passed_config2.syncope_tokens == ["a", "e"]
        assert passed_config2.intrusive_tokens == ["h", "'"]
        assert passed_config2.intrusive_max_stride == 2
        assert mock_prep.call_args[1]["enforce_phonotactics"] is True


def test_ctc_aligner_padded_midpoint_boundaries():
    """
    Verify that CTCSegmentationAligner.align pads chunk boundaries and resolves
    adjacent inter-chunk boundaries at the midpoint between chunks.
    """
    from transcription.alignment.models import TextChunk

    model = DummyASRModel()
    aligner = CTCSegmentationAligner(
        model=cast(Any, model),
        config=CTCAlignerConfig(boundary_pad_sec=0.1),
    )

    # 2 chunks: chunk 1 [0.5 - 1.0], chunk 2 [2.0 - 2.5]
    fake_timings = np.full(150, -1.0, dtype=np.float32)
    fake_timings[2] = 0.50
    fake_timings[4] = 2.00

    fake_char_probs = np.full(150, -0.1, dtype=np.float32)
    fake_state_list = ["a"] * 150
    fake_segments = [(0.5, 1.0, 0.05), (2.0, 2.5, 0.05)]

    chunks = [
        TextChunk(chunk_id="c1", text="a"),
        TextChunk(chunk_id="c2", text="a"),
    ]

    with (
        patch(
            "transcription.alignment.ctc_aligner.ctc_segmentation",
            return_value=(fake_timings, fake_char_probs, fake_state_list),
        ),
        patch(
            "transcription.alignment.ctc_aligner.determine_utterance_segments",
            return_value=fake_segments,
        ),
    ):
        audio = np.zeros(16000 * 4, dtype=np.float32)  # 4.0s total audio
        res = aligner.align(audio, chunks=chunks, cache=False)

        assert len(res.aligned_chunks) == 2
        c1 = res.aligned_chunks[0]
        c2 = res.aligned_chunks[1]

        # First chunk start should be padded: max(0.0, 0.5 - 0.1) = 0.4s
        assert c1.start_sec == 0.4

        # Midpoint between c1 word end (0.52) and c2 word start (2.0) is 1.26s
        assert c1.end_sec == 1.26
        assert c2.start_sec == 1.26

        # Last chunk end should be padded: min(4.0, 2.02 + 0.1) = 2.12s
        assert c2.end_sec == 2.12

        # Both chunks strictly cover their words
        assert c1.start_sec <= c1.words[0].start_sec
        assert c1.end_sec >= c1.words[-1].end_sec
        assert c2.start_sec <= c2.words[0].start_sec
        assert c2.end_sec >= c2.words[-1].end_sec


def test_syncope_class_prevents_kwo_vowel_clipping():
    """
    Verify that when the acoustic model outputs 'u' with high probability (argmax)
    for a word-final 'kwo' (e.g. 'nahskwo' in Mark 2:28 pronounced 'nahskwu'):
    1. In the logits around the final syllable, 'u' is likely and probability of 'u' is argmax.
    2. Flat syncope configuration (unpooled vowels) clips 'nahskwo' to 'nhskw' because
       the contrastive gate compares blank against near-zero 'o' probability.
    3. Syncope token class pooling (('a', 'e', 'i', 'o', 'u', 'v'), 't') pools vowel evidence,
       keeping the syncope gate closed and preserving the ground-truth word 'nahskwo'.
    """
    from ctc_segmentation import (  # type: ignore
        CtcSegmentationParameters,
        ctc_segmentation,
    )

    vocab = {
        "|": 0,
        "'": 1,
        "a": 2,
        "e": 3,
        "h": 4,
        "i": 5,
        "k": 6,
        "l": 7,
        "m": 8,
        "n": 9,
        "o": 10,
        "s": 11,
        "t": 12,
        "u": 13,
        "v": 14,
        "w": 15,
        "y": 16,
        "[UNK]": 17,
        "[PAD]": 18,
    }
    inv_vocab = {v: k for k, v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(vocab))]
    pad_id = vocab["[PAD]"]
    u_idx = vocab["u"]
    o_idx = vocab["o"]

    npz_path = Path("runs/cache/ctc_emissions/mark_02_e16d242d3b719b06.npz")
    if npz_path.exists():
        data = np.load(npz_path)
        lpz = data["lpz"][23250:23750]  # Mark 2:28 slice
    else:
        # Fallback synthetic logits
        T = 500
        lpz = np.full((T, len(vocab)), -15.0, dtype=np.float32)
        lpz[:, pad_id] = 0.0

    # Locate the vowel frame around the final syllable of 'nahskwo' in the slice (frame 258)
    vowel_frame = 258
    assert int(np.argmax(lpz[vowel_frame])) == u_idx
    assert lpz[vowel_frame, u_idx] > -0.05  # 'u' is highly likely (p > 0.95)
    assert lpz[vowel_frame, o_idx] < -10.0  # 'o' has near-zero log-posterior

    verse_text = [
        "nahski",
        "ihyvno",
        "yvwi",
        "uwetsi",
        "nahskwo",
        "ukvwiyuhsv",
        "unolvhitvhi",
    ]

    # 1. Flat syncope configuration (vowels unpooled):
    params_flat = CtcSegmentationParameters(
        char_list=char_list,
        blank=pad_id,
        syncope_tokens=["a", "e", "i", "o", "u", "v", "t"],
        intrusive_tokens=["h", "'"],
        intrusive_max_stride=4,
        index_duration=0.02,
        score_min_mean_over_L=2,
        replace_spaces_with_blanks=False,
    )
    gt_mat1, utt_indices1 = prepare_cherokee_text(params_flat, verse_text, char_list)
    timings1, char_probs1, state_list1 = ctc_segmentation(params_flat, lpz, gt_mat1)

    aligner = CTCSegmentationAligner()
    res_flat = aligner._extract_word_intervals(
        words=verse_text,
        timings=timings1,
        char_probs=char_probs1,
        state_list=state_list1,
        utt_indices=utt_indices1,
        start_word_idx=0,
        dur_sec=10.0,
        lead_offset_sec=0.0,
    )
    nahskwo_flat = next(w for w in res_flat if w.word == "nahskwo")
    # Flat syncope erroneously clips 'o' to 'kw'
    assert nahskwo_flat.emitted_word == "nhskw"

    # 2. Class-pooled syncope configuration:
    params_class = CtcSegmentationParameters(
        char_list=char_list,
        blank=pad_id,
        syncope_tokens=[["a", "e", "i", "o", "u", "v"], "t"],
        intrusive_tokens=["h", "'"],
        intrusive_max_stride=4,
        index_duration=0.02,
        score_min_mean_over_L=2,
        replace_spaces_with_blanks=False,
    )
    gt_mat2, utt_indices2 = prepare_cherokee_text(params_class, verse_text, char_list)
    timings2, char_probs2, state_list2 = ctc_segmentation(params_class, lpz, gt_mat2)

    res_class = aligner._extract_word_intervals(
        words=verse_text,
        timings=timings2,
        char_probs=char_probs2,
        state_list=state_list2,
        utt_indices=utt_indices2,
        start_word_idx=0,
        dur_sec=10.0,
        lead_offset_sec=0.0,
    )
    nahskwo_class = next(w for w in res_class if w.word == "nahskwo")
    # Class syncope pooling retains 'o' on canonical path
    assert nahskwo_class.emitted_word == "nahskwo"


def test_extract_word_intervals_vectorized_confidence():
    """Verify vectorized _extract_word_intervals correctly computes peaks, confidence, and min_char_prob."""
    aligner = CTCSegmentationAligner(
        config=CTCAlignerConfig(flag_min_confidence=0.5, flag_min_char_confidence=0.4)
    )
    words = ["test", "word"]
    # timings: for word 0, char_start_idx:end_idx = 1:5; for word 1, 6:10
    timings = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8, 0.9])
    utt_indices = [0, 5, 10]
    # state_list has 50 frames
    state_list = ["ε"] * 50
    # word 0 occupies frames 5 to 20 (0.1s to 0.4s + 0.02s => frames 5 to 21)
    state_list[7] = "t"
    state_list[12] = "e"
    state_list[16] = "s"
    state_list[19] = "t"

    # log-probabilities: np.log([0.9, 0.8, 0.7, 0.6])
    char_probs = np.full(50, -2.0, dtype=np.float32)
    # peaks for t, e, s, t:
    char_probs[7] = float(np.log(0.9))
    char_probs[12] = float(np.log(0.8))
    char_probs[16] = float(np.log(0.7))
    char_probs[19] = float(np.log(0.6))

    intervals = aligner._extract_word_intervals(
        words=words,
        timings=timings,
        char_probs=char_probs,
        state_list=state_list,
        utt_indices=utt_indices,
        start_word_idx=0,
        dur_sec=2.0,
    )

    assert len(intervals) == 2
    w0 = intervals[0]
    assert w0.word == "test"
    assert w0.emitted_word == "test"
    assert w0.flagged is False
    expected_peaks = [np.log(0.9), np.log(0.8), np.log(0.7), np.log(0.6)]
    expected_word_conf = float(np.exp(np.mean(expected_peaks)))
    expected_min_char = 0.6
    assert np.isclose(w0.confidence, expected_word_conf, atol=1e-4)
    assert w0.min_char_confidence is not None
    assert np.isclose(w0.min_char_confidence, expected_min_char, atol=1e-4)


def test_ctc_aligner_codeswitched_masks_forwarding(dummy_audio_file: Path):
    """
    Verify that CTCSegmentationAligner.align extracts code-switched token metadata
    from source_metadata and forwards custom token masks into prepare_cherokee_text.
    """
    from unittest.mock import MagicMock, patch
    from transcription.alignment.models import TextChunk
    from transcription.alignment.arpabet import (
        create_groundtruth_for_code_switched_syllabary,
    )

    aligner = CTCSegmentationAligner()

    line = "Guy Soldier: ᎯᎠ coffee ᎠᎩᏚᎵ"
    cs_res = create_groundtruth_for_code_switched_syllabary(line, strip_speaker=True)
    chunks = [TextChunk(chunk_id="chunk_01", text=cs_res.unified_tth)]
    source_metadata = {
        "chunk_01": {
            "text": line,
            "code_switched": cs_res.to_dict(),
        }
    }

    mock_model = MagicMock()
    mock_model.vocab = {
        "<pad>": 0,
        "a": 1,
        "e": 2,
        "i": 3,
        "o": 4,
        "u": 5,
        "v": 6,
        "k": 7,
        "h": 8,
        "s": 9,
        "t": 10,
        "l": 11,
        "'": 12,
        "|": 13,
    }
    mock_model.pad_token = "<pad>"
    mock_model.word_delimiter_token = "|"
    mock_model.decode.return_value = MagicMock(
        text="hi'a khasi akituli", confidence=0.95
    )

    fake_lpz = np.zeros((100, len(mock_model.vocab)), dtype=np.float32)

    with patch.object(
        aligner, "get_logits_cached", return_value=(fake_lpz, 2.0, 16000)
    ):
        with patch(
            "transcription.alignment.ctc_aligner.prepare_cherokee_text",
            wraps=prepare_cherokee_text,
        ) as mock_prep:
            with patch(
                "transcription.alignment.ctc_aligner.ctc_segmentation"
            ) as mock_ctc:
                # Mock ctc_segmentation returns
                mock_ctc.return_value = (
                    np.zeros(20),
                    np.zeros(100),
                    ["ε"] * 100,
                )
                aligner.align(
                    dummy_audio_file,
                    chunks=chunks,
                    source_id="test_audio",
                    asr_model=mock_model,
                    source_metadata=source_metadata,
                )

                assert mock_prep.called
                _, kwargs = mock_prep.call_args
                assert "token_masks" in kwargs
                token_masks = kwargs["token_masks"]
                assert token_masks is not None
                # Word 0: hi'a (Cherokee), Word 1: khasi (English coffee -> zero masks), Word 2: akituli (Cherokee)
                # Word 1 (khasi) must have all False for syncope and intrusion
                khasi_sync, khasi_intrus = token_masks[1]
                assert not any(khasi_sync)
                assert not any(khasi_intrus)


def test_ctc_aligner_with_vad_soft_masking(tmp_path: Path):
    """Verify that CTCSegmentationAligner invokes mask_non_speech_logits when enable_vad_soft_masking=True."""
    dummy_audio_file = tmp_path / "dummy.wav"
    audio = AudioSegment.silent(duration=2000, frame_rate=16000)
    audio.export(dummy_audio_file, format="wav")

    config = CTCAlignerConfig(
        enable_vad_soft_masking=True,
        vad_p_low=0.15,
        vad_p_high=0.60,
    )
    aligner = CTCSegmentationAligner(config=config)
    chunks = [TextChunk(chunk_id="c1", text="tsisa kalonetv")]

    mock_model = MagicMock()
    mock_model.processor.tokenizer.get_vocab.return_value = {
        "<pad>": 0,
        "t": 1,
        "s": 2,
        "i": 3,
        "a": 4,
        "k": 5,
        "l": 6,
        "o": 7,
        "n": 8,
        "e": 9,
        "v": 10,
        "|": 11,
    }
    mock_model.processor.tokenizer.pad_token_id = 0

    fake_lpz = np.zeros((100, 12), dtype=np.float32)

    with patch.object(
        aligner, "get_logits_cached", return_value=(fake_lpz, 2.0, 16000)
    ):
        with patch(
            "transcription.alignment.ctc_aligner.mask_non_speech_logits",
            wraps=lambda lpz, **kwargs: lpz,
        ) as mock_mask:
            with patch(
                "transcription.alignment.ctc_aligner.ctc_segmentation"
            ) as mock_ctc:
                mock_ctc.return_value = (
                    np.zeros(20),
                    np.zeros(100),
                    ["ε"] * 100,
                )
                aligner.align(
                    dummy_audio_file,
                    chunks=chunks,
                    source_id="test_audio",
                    asr_model=mock_model,
                )
                assert mock_mask.called
                _, kwargs = mock_mask.call_args
                assert kwargs["p_low"] == 0.15
                assert kwargs["p_high"] == 0.60
