# -*- coding: utf-8 -*-
"""
Unit tests for audio perturbation transforms in transcription.evaluation.perturbations.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from transcription.evaluation.perturbations import (
    AcousticFilter,
    AdditiveNoise,
    AudioTransform,
    ComposeTransforms,
    TimeWarp,
)


class TestAudioTransformABC:
    """Test AudioTransform abstract base class behaviors."""

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            AudioTransform()  # type: ignore


class TestAdditiveNoise:
    """Tests for AdditiveNoise perturbation transform."""

    def test_invalid_noise_type(self):
        with pytest.raises(ValueError, match="Unsupported noise_type"):
            AdditiveNoise(snr_db=10.0, noise_type="invalid_type")  # type: ignore

    def test_white_noise_snr(self):
        torch.manual_seed(42)
        sr = 16000
        # 1-second pure tone at 440 Hz
        t = torch.arange(sr, dtype=torch.float32) / sr
        waveform = 0.5 * torch.sin(2 * np.pi * 440.0 * t).unsqueeze(0)  # [1, 16000]

        target_snr = 15.0
        transform = AdditiveNoise(snr_db=target_snr, noise_type="white")
        noisy = transform(waveform, sample_rate=sr)

        assert noisy.shape == waveform.shape
        # Check clamping
        assert torch.all(noisy >= -1.0) and torch.all(noisy <= 1.0)

        # Measure noise component power
        noise_component = noisy - waveform
        px = torch.mean(waveform**2).item()
        p_noise = torch.mean(noise_component**2).item()
        measured_snr = 10.0 * math.log10(px / p_noise)
        assert abs(measured_snr - target_snr) < 0.5

    def test_pink_noise_snr_and_shape(self):
        torch.manual_seed(42)
        sr = 16000
        t = torch.arange(sr, dtype=torch.float32) / sr
        waveform = 0.6 * torch.cos(2 * np.pi * 300.0 * t)  # 1D tensor [16000]

        target_snr = 10.0
        transform = AdditiveNoise(snr_db=target_snr, noise_type="pink")
        noisy = transform(waveform, sample_rate=sr)

        # Ensure 1D shape is preserved
        assert noisy.shape == waveform.shape
        assert noisy.dim() == 1

        noise_component = noisy - waveform
        px = torch.mean(waveform**2).item()
        p_noise = torch.mean(noise_component**2).item()
        measured_snr = 10.0 * math.log10(px / p_noise)
        assert abs(measured_snr - target_snr) < 0.5

    def test_ambient_noise_with_file_looping_and_resampling(self):
        sr = 16000
        ambient_sr = 8000
        # Create a short 0.25s ambient audio file (shorter than target 1.0s to test looping)
        short_ambient = (
            np.random.randn(int(ambient_sr * 0.25), 1).astype(np.float32) * 0.1
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, short_ambient, ambient_sr)

            transform = AdditiveNoise(
                snr_db=12.0, noise_type="ambient", ambient_path=temp_path
            )
            # 2-channel 1.0s waveform
            waveform = 0.4 * torch.randn(2, sr)
            noisy = transform(waveform, sample_rate=sr)

            assert noisy.shape == (2, sr)
            noise_component = noisy - waveform
            px = torch.mean(waveform**2).item()
            p_noise = torch.mean(noise_component**2).item()
            measured_snr = 10.0 * math.log10(px / p_noise)
            assert abs(measured_snr - 12.0) < 0.5
        finally:
            p = Path(temp_path)
            if p.exists():
                p.unlink()

    def test_ambient_noise_fallback_when_file_missing(self):
        sr = 16000
        waveform = 0.5 * torch.randn(1, sr)
        # Nonexistent ambient path
        transform = AdditiveNoise(
            snr_db=10.0,
            noise_type="ambient",
            ambient_path="/path/that/does/not/exist.wav",
        )
        noisy = transform(waveform, sample_rate=sr)
        assert noisy.shape == waveform.shape
        assert not torch.isnan(noisy).any()

    def test_zero_power_signal(self):
        waveform = torch.zeros(1, 16000)
        transform = AdditiveNoise(snr_db=10.0, noise_type="white")
        noisy = transform(waveform, sample_rate=16000)
        assert torch.equal(noisy, waveform)


class TestTimeWarp:
    """Tests for TimeWarp transform with pitch preservation."""

    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="must be strictly positive"):
            TimeWarp(rate=0.0)
        with pytest.raises(ValueError, match="must be strictly positive"):
            TimeWarp(rate=-1.0)

    def test_rate_one_identity(self):
        waveform = torch.randn(1, 16000)
        transform = TimeWarp(rate=1.0)
        out = transform(waveform, sample_rate=16000)
        assert torch.equal(out, waveform)

    def test_pitch_preservation(self):
        sr = 16000
        freq = 440.0  # A4
        t = torch.arange(sr, dtype=torch.float32) / sr
        waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)  # [1, 16000]

        # Time warp rate 1.15 (tempo speedup)
        transform = TimeWarp(rate=1.15)
        sped_up = transform(waveform, sample_rate=sr)

        # Length should decrease
        assert sped_up.shape[-1] < waveform.shape[-1]

        # Compute peak frequency of time-stretched signal via FFT
        fft_mag = torch.abs(torch.fft.rfft(sped_up[0]))
        freqs = torch.fft.rfftfreq(sped_up.shape[-1], d=1.0 / sr)
        peak_freq = freqs[torch.argmax(fft_mag)].item()

        # Pitch must be preserved within ~5 Hz
        assert (
            abs(peak_freq - freq) < 5.0
        ), f"Pitch was altered by TimeStretch! Expected {freq} Hz, got {peak_freq:.2f} Hz"

    def test_time_warp_slow_down(self):
        sr = 16000
        waveform = torch.randn(2, sr)
        transform = TimeWarp(rate=0.85)
        slowed = transform(waveform, sample_rate=sr)

        assert slowed.dim() == 2
        assert slowed.shape[0] == 2
        # Slower rate -> longer output
        assert slowed.shape[1] > waveform.shape[1]

    def test_time_warp_1d_tensor(self):
        waveform = torch.randn(8000)
        transform = TimeWarp(rate=1.1)
        out = transform(waveform, sample_rate=16000)
        assert out.dim() == 1
        assert out.shape[0] < waveform.shape[0]


class TestAcousticFilter:
    """Tests for AcousticFilter Butterworth bandpass filtering."""

    def test_invalid_cutoffs(self):
        with pytest.raises(ValueError, match="lowcut must be positive"):
            AcousticFilter(lowcut=-10.0, highcut=3400.0)
        with pytest.raises(ValueError, match="must be greater than"):
            AcousticFilter(lowcut=3500.0, highcut=3400.0)

    def test_bandpass_attenuation_and_passband(self):
        sr = 16000
        t = torch.arange(sr, dtype=torch.float32) / sr

        # 100 Hz (below 300 Hz cutoff)
        f_low = torch.sin(2 * np.pi * 100.0 * t)
        # 1000 Hz (within 300-3400 Hz passband)
        f_mid = torch.sin(2 * np.pi * 1000.0 * t)
        # 6000 Hz (above 3400 Hz cutoff)
        f_high = torch.sin(2 * np.pi * 6000.0 * t)

        filter_tf = AcousticFilter(lowcut=300.0, highcut=3400.0, order=4)

        y_low = filter_tf(f_low, sample_rate=sr)
        y_mid = filter_tf(f_mid, sample_rate=sr)
        y_high = filter_tf(f_high, sample_rate=sr)

        # Check shapes
        assert y_low.shape == f_low.shape
        assert y_mid.shape == f_mid.shape
        assert y_high.shape == f_high.shape

        # Calculate steady-state RMS (ignoring filter startup transient in first 1000 samples)
        rms_low = torch.sqrt(torch.mean(y_low[1000:] ** 2)).item()
        rms_mid = torch.sqrt(torch.mean(y_mid[1000:] ** 2)).item()
        rms_high = torch.sqrt(torch.mean(y_high[1000:] ** 2)).item()

        # Input sine RMS is 1 / sqrt(2) ~ 0.7071
        expected_pass_rms = 1.0 / np.sqrt(2)
        assert abs(rms_mid - expected_pass_rms) < 0.05  # Passband intact
        assert rms_low < 0.02  # Strongly attenuated (> 30 dB down)
        assert rms_high < 0.02  # Strongly attenuated (> 30 dB down)

    def test_filter_multichannel_and_nyquist_guard(self):
        sr = 8000  # Nyquist is 4000 Hz
        waveform = torch.randn(2, sr)
        # highcut close to or at Nyquist
        filter_tf = AcousticFilter(lowcut=300.0, highcut=3900.0, order=4)
        out = filter_tf(waveform, sample_rate=sr)
        assert out.shape == waveform.shape
        assert not torch.isnan(out).any()


class TestComposeTransforms:
    """Tests for ComposeTransforms sequential pipeline."""

    def test_compose_pipeline(self):
        sr = 16000
        waveform = 0.5 * torch.randn(1, sr)

        pipeline = ComposeTransforms(
            [
                AdditiveNoise(snr_db=20.0, noise_type="white"),
                AcousticFilter(lowcut=300.0, highcut=3400.0),
                TimeWarp(rate=1.1),
            ]
        )

        out = pipeline(waveform, sample_rate=sr)
        assert out.dim() == 2
        assert out.shape[0] == 1
        assert out.shape[1] < waveform.shape[1]
        assert not torch.isnan(out).any()
        assert "ComposeTransforms" in repr(pipeline)

    def test_float64_support(self):
        sr = 16000
        waveform = torch.randn(2, 8000, dtype=torch.float64)
        pipeline = ComposeTransforms(
            [
                AdditiveNoise(snr_db=15.0, noise_type="pink"),
                AcousticFilter(lowcut=300.0, highcut=3400.0),
                TimeWarp(rate=1.05),
            ]
        )
        out = pipeline(waveform, sample_rate=sr)
        assert out.dtype == torch.float64
        assert out.shape[0] == 2
        assert not torch.isnan(out).any()
