# -*- coding: utf-8 -*-
"""
audio_segmenter.py

Audio pre-segmentation module using VAD silence splitting logic from process_interviews.py.
Chunks long recordings into manageable pieces (<10s) while tracking global timestamp offsets.
"""

from dataclasses import dataclass
from typing import List, Union
from pydub import AudioSegment

from transcription.audio.segment import (
    get_energy_profile,
    segment_audio_from_profile,
    get_best_parameters,
    split_long_segments_smart,
)


@dataclass
class AudioChunk:
    chunk_index: int
    audio: AudioSegment
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


def segment_long_audio(
    audio_or_path: Union[str, AudioSegment],
    max_duration_ms: int = 10000,
    overlap_ms: int = 250,
) -> List[AudioChunk]:
    """
    Splits long audio into chunks <= max_duration_ms on silence/pause boundaries.

    Args:
        audio_or_path: File path (str) or loaded AudioSegment instance.
        max_duration_ms: Max chunk length in milliseconds (default: 10000ms = 10s).
        overlap_ms: Overlap padding in ms for fallback splits to prevent word truncation.

    Returns:
        List of AudioChunk objects containing segment audio and global start/end offsets (in seconds).
    """
    if isinstance(audio_or_path, str):
        audio = AudioSegment.from_file(audio_or_path)
    else:
        audio = audio_or_path

    total_len_ms = len(audio)
    dbfs_profile = get_energy_profile(audio, step_ms=10)
    best_config = get_best_parameters(dbfs_profile, total_len_ms)
    initial_segments = best_config.get("segments", [])

    if not initial_segments:
        # If no silence breaks were found, treat whole audio as one segment
        initial_segments = [{"start": 0, "end": len(audio), "duration": len(audio)}]

    # Step 2: Split any segments longer than max_duration_ms
    final_segments = []
    for seg in initial_segments:
        smart_splits = []
        # split_long_segments_smart appends to its internal list via closure helper,
        # so we pass an isolated sub-segment call
        sub_audio = audio[seg["start"] : seg["end"]]

        # Split segment if it exceeds max_duration_ms
        if seg["duration"] > max_duration_ms:
            smart_segs = []

            def _collect_splits(start_ms, end_ms):
                dur = end_ms - start_ms
                if dur <= max_duration_ms:
                    smart_segs.append(
                        {"start": start_ms, "end": end_ms, "duration": dur}
                    )
                    return

                # Split using quietest window
                from scripts.process_interviews import (
                    get_energy_profile,
                    segment_audio_from_profile,
                )
                import numpy as np

                sub_sub = sub_audio[start_ms:end_ms]
                dbfs = get_energy_profile(sub_sub, step_ms=10)
                sub_splits = segment_audio_from_profile(
                    dbfs, dur, step_ms=10, min_silence_len=200, silence_thresh=-30
                )
                if sub_splits and len(sub_splits) > 1:
                    for ss in sub_splits:
                        _collect_splits(start_ms + ss["start"], start_ms + ss["end"])
                    return

                # Fallback mid split
                mid_ms = dur // 2
                left_end = min(end_ms, start_ms + mid_ms + overlap_ms)
                right_start = max(start_ms, start_ms + mid_ms - overlap_ms)
                _collect_splits(start_ms, left_end)
                _collect_splits(right_start, end_ms)

            _collect_splits(0, len(sub_audio))
            for ss in smart_segs:
                final_segments.append(
                    {
                        "start": seg["start"] + ss["start"],
                        "end": seg["start"] + ss["end"],
                    }
                )
        else:
            final_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )

    # Step 3: Package into AudioChunk data structures with global second offsets
    chunks = []
    for idx, seg in enumerate(final_segments):
        chunk_audio = audio[seg["start"] : seg["end"]]
        start_sec = round(seg["start"] / 1000.0, 3)
        end_sec = round(seg["end"] / 1000.0, 3)
        chunks.append(
            AudioChunk(
                chunk_index=idx,
                audio=chunk_audio,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    return chunks
