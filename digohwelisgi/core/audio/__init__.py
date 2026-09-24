# Core audio processing, segmentation, and VAD masking
from digohwelisgi.core.audio.segment import (
    AudioChunk,
    get_energy_profile,
    segment_audio_from_profile,
    compute_metrics,
    get_best_parameters,
    split_long_segments_smart,
    segment_long_audio,
)
from digohwelisgi.core.audio.masking import (
    SileroVADDetector,
    mask_non_speech_logits,
    apply_vad_soft_masking,
    extract_vad_intervals,
)

TARGET_SAMPLE_RATE = 16000

__all__ = [
    "TARGET_SAMPLE_RATE",
    "AudioChunk",
    "get_energy_profile",
    "segment_audio_from_profile",
    "compute_metrics",
    "get_best_parameters",
    "split_long_segments_smart",
    "segment_long_audio",
    "SileroVADDetector",
    "mask_non_speech_logits",
    "apply_vad_soft_masking",
    "extract_vad_intervals",
]
