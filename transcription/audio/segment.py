# -*- coding: utf-8 -*-
"""
transcription.audio.segment backward-compatibility / re-export shim to transcription.core.audio.segment.
"""

from transcription.core.audio.segment import (
    AudioChunk,
    get_energy_profile,
    segment_audio_from_profile,
    compute_metrics,
    print_table,
    get_best_parameters,
    split_long_segments_smart,
    segment_long_audio,
    main,
)

__all__ = [
    "AudioChunk",
    "get_energy_profile",
    "segment_audio_from_profile",
    "compute_metrics",
    "print_table",
    "get_best_parameters",
    "split_long_segments_smart",
    "segment_long_audio",
    "main",
]

if __name__ == "__main__":
    main()
