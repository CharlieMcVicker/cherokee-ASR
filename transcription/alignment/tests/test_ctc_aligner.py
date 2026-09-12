# -*- coding: utf-8 -*-
"""
test_ctc_aligner.py

Unit tests for CTCSegmentationAligner disk caching, deterministic cache keys,
and get_logits_cached helper.
"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
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
        num_frames = max(10, len(samples) // 320)
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
    # Create a 0.5-second 16kHz silent audio file
    silence = AudioSegment.silent(duration=500, frame_rate=16000)
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
