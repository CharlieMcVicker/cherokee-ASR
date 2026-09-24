# -*- coding: utf-8 -*-
"""
test_audio_segmenter.py

Unit tests for audio segmentation in digohwelisgi.core.audio.segment.
"""

import unittest
from pydub import AudioSegment
from pydub.generators import Sine

from digohwelisgi.core.audio.segment import (
    segment_long_audio,
    AudioChunk,
    get_energy_profile,
)


class TestAudioSegmenter(unittest.TestCase):
    def test_segment_synthetic_audio(self):
        # Generate 12 seconds of audio with silence in the middle
        tone1 = Sine(440).to_audio_segment(duration=5000)
        silence = AudioSegment.silent(duration=2000)
        tone2 = Sine(880).to_audio_segment(duration=5000)

        audio = tone1 + silence + tone2
        self.assertEqual(len(audio), 12000)

        chunks = segment_long_audio(audio, max_duration_ms=6000)
        self.assertTrue(len(chunks) >= 2)
        self.assertIsInstance(chunks[0], AudioChunk)
        self.assertGreaterEqual(chunks[0].end_sec, chunks[0].start_sec)
        self.assertEqual(chunks[0].start_sec, 0.0)

    def test_energy_profile(self):
        tone = Sine(440).to_audio_segment(duration=1000)
        profile = get_energy_profile(tone, step_ms=10)
        self.assertEqual(len(profile), 100)
        self.assertTrue(all(profile > -60.0))


if __name__ == "__main__":
    unittest.main()
