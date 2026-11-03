import numpy as np
import pytest
from digohwelisgi.core.audio.augmentation import (
    AudioAugmentationConfig,
    AudioAugmenter,
    create_audio_augmenter,
)


def test_default_config():
    config = AudioAugmentationConfig()
    assert config.time_stretch_p == 0.5
    assert config.time_stretch_min_rate == 0.9
    assert config.time_stretch_max_rate == 1.1
    assert config.pitch_shift_p == 0.5
    assert config.gaussian_noise_p == 0.3


def test_no_op_augmentation():
    config = AudioAugmentationConfig(
        time_stretch_p=0.0,
        pitch_shift_p=0.0,
        gaussian_noise_p=0.0,
    )
    augmenter = AudioAugmenter(config=config)
    audio = np.ones(16000, dtype=np.float32) * 0.5
    augmented = augmenter(audio, sample_rate=16000)

    np.testing.assert_allclose(augmented, audio, atol=1e-6)


def test_gaussian_noise_augmentation():
    config = AudioAugmentationConfig(
        time_stretch_p=0.0,
        pitch_shift_p=0.0,
        gaussian_noise_p=1.0,
        gaussian_noise_min_amplitude=0.01,
        gaussian_noise_max_amplitude=0.05,
    )
    augmenter = AudioAugmenter(config=config)
    audio = np.zeros(16000, dtype=np.float32)
    augmented = augmenter(audio, sample_rate=16000)

    assert augmented.shape == audio.shape
    assert augmented.dtype == np.float32
    assert not np.allclose(augmented, audio)
    assert np.std(augmented) > 0.005


def test_time_stretch_and_pitch_shift_augmentation():
    config = AudioAugmentationConfig(
        time_stretch_p=1.0,
        time_stretch_min_rate=0.9,
        time_stretch_max_rate=1.1,
        time_stretch_leave_length_unchanged=True,
        pitch_shift_p=1.0,
        pitch_shift_min_semitones=-2.0,
        pitch_shift_max_semitones=2.0,
        gaussian_noise_p=0.0,
    )
    augmenter = create_audio_augmenter(config=config)
    # Generate simple sine wave
    t = np.linspace(0, 1, 16000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    augmented = augmenter(audio, sample_rate=16000)

    assert augmented.dtype == np.float32
    assert len(augmented) == len(audio)
    assert not np.allclose(augmented, audio)
