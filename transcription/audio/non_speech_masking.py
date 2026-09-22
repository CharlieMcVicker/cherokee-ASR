# -*- coding: utf-8 -*-
"""
transcription.audio.non_speech_masking backward-compatibility / re-export shim to transcription.core.audio.masking.
"""

from transcription.core.audio.masking import (
    SILERO_VAD_URL,
    DEFAULT_VAD_CACHE_PATH,
    SileroVADDetector,
    _load_audio_as_16k_tensor,
    mask_non_speech_logits,
    apply_vad_soft_masking,
    extract_vad_intervals,
)

__all__ = [
    "SILERO_VAD_URL",
    "DEFAULT_VAD_CACHE_PATH",
    "SileroVADDetector",
    "_load_audio_as_16k_tensor",
    "mask_non_speech_logits",
    "apply_vad_soft_masking",
    "extract_vad_intervals",
]
