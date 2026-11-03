"""Audio data augmentation module for training pipelines.

Provides language-agnostic audio augmentation transforms (time-stretching,
pitch-shifting, and Gaussian noise injection) using audiomentations.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from audiomentations import AddGaussianNoise, Compose, PitchShift, TimeStretch


@dataclass(frozen=True)
class AudioAugmentationConfig:
    """Configuration parameters for audio augmentation transforms."""

    time_stretch_min_rate: float = 0.9
    time_stretch_max_rate: float = 1.1
    time_stretch_leave_length_unchanged: bool = True
    time_stretch_p: float = 0.5

    pitch_shift_min_semitones: float = -2.0
    pitch_shift_max_semitones: float = 2.0
    pitch_shift_p: float = 0.5

    gaussian_noise_min_amplitude: float = 0.001
    gaussian_noise_max_amplitude: float = 0.01
    gaussian_noise_p: float = 0.3


def build_audiomentations_pipeline(config: AudioAugmentationConfig) -> Compose:
    """Build an audiomentations Compose pipeline from configuration."""
    transforms = []

    if config.time_stretch_p > 0.0:
        transforms.append(
            TimeStretch(
                min_rate=config.time_stretch_min_rate,
                max_rate=config.time_stretch_max_rate,
                leave_length_unchanged=config.time_stretch_leave_length_unchanged,
                p=config.time_stretch_p,
            )
        )

    if config.pitch_shift_p > 0.0:
        transforms.append(
            PitchShift(
                min_semitones=config.pitch_shift_min_semitones,
                max_semitones=config.pitch_shift_max_semitones,
                p=config.pitch_shift_p,
            )
        )

    if config.gaussian_noise_p > 0.0:
        transforms.append(
            AddGaussianNoise(
                min_amplitude=config.gaussian_noise_min_amplitude,
                max_amplitude=config.gaussian_noise_max_amplitude,
                p=config.gaussian_noise_p,
            )
        )

    return Compose(transforms)


class AudioAugmenter:
    """Stateless audio augmenter applying configured transformations."""

    def __init__(
        self,
        config: Optional[AudioAugmentationConfig] = None,
        pipeline: Optional[Compose] = None,
    ) -> None:
        self.config = config or AudioAugmentationConfig()
        self.pipeline = pipeline or build_audiomentations_pipeline(self.config)

    def __call__(self, samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply audio augmentation transforms to a 1D waveform array."""
        samples_f32 = np.asarray(samples, dtype=np.float32)
        return self.pipeline(samples=samples_f32, sample_rate=sample_rate)


def create_audio_augmenter(
    config: Optional[AudioAugmentationConfig] = None,
) -> AudioAugmenter:
    """Factory helper to construct an AudioAugmenter instance."""
    return AudioAugmenter(config=config)
