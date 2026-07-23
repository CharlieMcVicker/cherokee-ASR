# -*- coding: utf-8 -*-
"""
test_exporter.py

Unit tests for exporter.py
"""

import unittest
import os
import tempfile
from transcription.timestamping.aligner import (
    AlignmentResult,
    VerseInterval,
    WordInterval,
)
from transcription.timestamping.exporter import (
    export_praat_textgrid,
    export_alignment_manifest,
)


class TestExporter(unittest.TestCase):
    def setUp(self):
        w1 = WordInterval(
            word="ataleniskv", start_sec=1.0, end_sec=1.5, confidence=0.95
        )
        w2 = WordInterval(word="yisthv", start_sec=1.6, end_sec=2.0, confidence=0.90)
        v1 = VerseInterval(
            line_id="020101",
            cherokee_syllabary="ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ",
            raw_phonetic="A-da-le-ni-s-gv yi-s-dv",
            english="The beginning",
            start_sec=1.0,
            end_sec=2.0,
            words=[w1, w2],
        )
        self.alignment = AlignmentResult(audio_source="mark_01.wav", verses=[v1])

    def test_export_praat_textgrid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test.TextGrid")
            export_praat_textgrid(self.alignment, out_file)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r") as f:
                content = f.read()
            self.assertIn("ooTextFile", content)
            self.assertIn("Verses", content)
            self.assertIn("Words", content)
            self.assertIn("Padded Words", content)
            self.assertIn("Raw ASR Emissions", content)
            self.assertIn("size = 4", content)

    def test_padded_words_overlap_fusion(self):
        w1 = WordInterval(word="word1", start_sec=1.000, end_sec=1.050)
        w2 = WordInterval(
            word="word2", start_sec=1.055, end_sec=1.100
        )  # w1+10ms=1.060 > 1.055 -> overlaps w2
        w3 = WordInterval(word="word3", start_sec=1.500, end_sec=1.600)
        v1 = VerseInterval(
            line_id="010101",
            cherokee_syllabary="w1 w2 w3",
            raw_phonetic="word1 word2 word3",
            english="test",
            start_sec=1.0,
            end_sec=2.0,
            words=[w1, w2, w3],
        )
        alignment = AlignmentResult(audio_source="test.wav", verses=[v1])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "padded.TextGrid")
            export_praat_textgrid(alignment, out_file)
            with open(out_file, "r") as f:
                content = f.read()
            self.assertIn('"word1 word2"', content)
            self.assertIn('"word3"', content)

    def test_export_alignment_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "alignment_manifest.json")
            export_alignment_manifest(self.alignment, out_file)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r") as f:
                content = f.read()
            self.assertIn("mark_01.wav", content)
            self.assertIn("020101", content)


if __name__ == "__main__":
    unittest.main()
