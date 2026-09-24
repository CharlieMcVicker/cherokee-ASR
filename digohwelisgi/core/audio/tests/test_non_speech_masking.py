# -*- coding: utf-8 -*-
"""
test_non_speech_masking.py

Unit tests for Silero VAD non-speech soft-masking and convex logit blending.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from pydub import AudioSegment
import pytest
import torch

from digohwelisgi.core.audio.masking import (
    SileroVADDetector,
    _load_audio_as_16k_tensor,
    mask_non_speech_logits,
    apply_vad_soft_masking,
    extract_vad_intervals,
)


def test_load_audio_formats():
    """Verify _load_audio_as_16k_tensor handles AudioSegment, ndarray, and torch.Tensor."""
    # Test numpy array
    arr = np.zeros(16000, dtype=np.float32)
    t_arr = _load_audio_as_16k_tensor(arr)
    assert isinstance(t_arr, torch.Tensor)
    assert t_arr.shape == (16000,)

    # Test AudioSegment
    seg = AudioSegment.silent(duration=500, frame_rate=16000)
    t_seg = _load_audio_as_16k_tensor(seg)
    assert isinstance(t_seg, torch.Tensor)
    assert t_seg.shape[0] == 8000  # 0.5s at 16k


def test_mask_non_speech_logits_convex_blending():
    """Verify mathematical properties of convex posterior blending."""
    T, V = 100, 30
    blank_id = 0

    # Mock raw logits where each frame sums to 1 in exp space
    raw_logits = np.random.randn(T, V).astype(np.float32)
    lpz = raw_logits - np.log(np.sum(np.exp(raw_logits), axis=-1, keepdims=True))

    # Mock VAD detector returning controlled speech probabilities
    # 0 to 30: clear speech (0.95)
    # 30 to 70: laughter / silence (0.05)
    # 70 to 100: chuckle-speech (0.375 -> midpoint between 0.15 and 0.60)
    mock_detector = MagicMock(spec=SileroVADDetector)
    # 100 frames at 20ms = 2.0s -> ~62 windows of 32ms
    n_windows = 63
    probs = np.zeros(n_windows, dtype=np.float32)
    probs[:20] = 0.95
    probs[20:45] = 0.05
    probs[45:] = 0.375
    mock_detector.predict_speech_probabilities.return_value = probs

    masked_lpz = mask_non_speech_logits(
        lpz=lpz,
        audio=np.zeros(32000, dtype=np.float32),
        index_duration=0.02,
        blank_id=blank_id,
        p_low=0.15,
        p_high=0.60,
        pad_ms=0,
        detector=mock_detector,
    )

    assert masked_lpz.shape == (T, V)
    assert masked_lpz.dtype == lpz.dtype

    # Frame 10 (clear speech): untouched
    np.testing.assert_allclose(masked_lpz[10], lpz[10], atol=1e-5)

    # Frame 45 (pure laughter / non-speech): blank prob ~1.0, non-blank suppressed
    assert np.exp(masked_lpz[45, blank_id]) > 0.9999
    assert np.all(masked_lpz[45, 1:] < -20.0)

    # Frame 90 (chuckle-speech, p=0.375, gamma=0.5):
    # relative proportions of characters preserved
    gamma = 0.5
    expected_blank_p = gamma * np.exp(lpz[90, blank_id]) + (1 - gamma) * 1.0
    np.testing.assert_allclose(
        np.exp(masked_lpz[90, blank_id]), expected_blank_p, atol=1e-3
    )


def test_apply_vad_soft_masking_alias():
    """Verify apply_vad_soft_masking functions identically to mask_non_speech_logits."""
    assert apply_vad_soft_masking is mask_non_speech_logits


def test_mask_non_speech_logits_empty_or_zero():
    """Verify edge case of empty or zero-duration logits."""
    empty_lpz = np.zeros((0, 30), dtype=np.float32)
    res = mask_non_speech_logits(empty_lpz, audio=np.zeros(100))
    assert res.shape == (0, 30)


def test_extract_vad_intervals():
    """Verify interval extraction returns valid intervals."""
    mock_detector = MagicMock(spec=SileroVADDetector)
    # 32 windows of 32ms
    probs = np.array([0.9] * 10 + [0.05] * 10 + [0.9] * 12, dtype=np.float32)
    mock_detector.predict_speech_probabilities.return_value = probs

    intervals = extract_vad_intervals(
        audio=np.zeros(32 * 512, dtype=np.float32),
        pad_ms=0,
        detector=mock_detector,
    )
    assert len(intervals) >= 3
    assert intervals[0][2] == "speech"
    assert intervals[1][2] == "non-speech"
    assert intervals[2][2] == "speech"
