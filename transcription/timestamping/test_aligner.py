# -*- coding: utf-8 -*-
"""
test_aligner.py

Unit tests for aligner.py
"""

import unittest
from transcription.timestamping.aligner import align_tokens_to_verses, AlignmentResult


class TestAligner(unittest.TestCase):
    def test_align_tokens_to_verses(self):
        tokens = [
            {
                "word": "ataleniskv",
                "start_time": 1.0,
                "end_time": 1.5,
                "confidence": 0.95,
            },
            {"word": "yisthv", "start_time": 1.6, "end_time": 2.0, "confidence": 0.90},
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ",
                "raw_phonetic": "A-da-le-ni-s-gv yi-s-dv",
                "normalized_text": "ataleniskv yisthv",
                "english": "The beginning of the gospel",
            }
        ]

        result = align_tokens_to_verses(tokens, verses, audio_source="mark_01.wav")
        self.assertIsInstance(result, AlignmentResult)
        self.assertEqual(len(result.verses), 1)
        v0 = result.verses[0]
        self.assertEqual(v0.line_id, "020101")
        self.assertEqual(v0.start_sec, 1.0)
        self.assertEqual(v0.end_sec, 2.0)
        self.assertEqual(len(v0.words), 2)

    def test_preamble_skip(self):
        # Tokens include leading preamble intro chatter (0.0s - 2.5s)
        tokens = [
            {"word": "mark", "start_time": 0.1, "end_time": 0.5, "confidence": 0.90},
            {"word": "chapter", "start_time": 0.6, "end_time": 1.1, "confidence": 0.90},
            {"word": "one", "start_time": 1.2, "end_time": 1.8, "confidence": 0.90},
            # Actual verse start at 2.5s
            {
                "word": "ataleniskv",
                "start_time": 2.5,
                "end_time": 3.0,
                "confidence": 0.95,
            },
            {"word": "yisthv", "start_time": 3.1, "end_time": 3.5, "confidence": 0.90},
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ",
                "raw_phonetic": "A-da-le-ni-s-gv yi-s-dv",
                "normalized_text": "ataleniskv yisthv",
                "english": "The beginning of the gospel",
            }
        ]

        result = align_tokens_to_verses(tokens, verses, audio_source="mark_01.wav")
        v0 = result.verses[0]
        # Preamble tokens ("mark chapter one") skipped; start_sec correctly lands at 2.5s
        self.assertEqual(v0.start_sec, 2.5)
        self.assertEqual(v0.end_sec, 3.5)

    def test_word_fusion_with_glottal_or_h(self):
        # A single transcribed token merges two ground truth words with an 'h' or glottal insertion
        tokens = [
            {
                "word": "ataleniskvhyisthv",  # merged pair
                "start_time": 1.0,
                "end_time": 2.5,
                "confidence": 0.92,
            }
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ",
                "raw_phonetic": "A-da-le-ni-s-gv yi-s-dv",
                "normalized_text": "ataleniskv yisthv",
                "english": "The beginning of the gospel",
            }
        ]

        result = align_tokens_to_verses(tokens, verses, audio_source="mark_01.wav")
        v0 = result.verses[0]
        self.assertEqual(len(v0.words), 1)
        self.assertEqual(v0.words[0].word, "A-da-le-ni-s-gv yi-s-dv")
        self.assertEqual(v0.words[0].start_sec, 1.0)
        self.assertEqual(v0.words[0].end_sec, 2.5)

    def test_non_overlapping_word_timestamps(self):
        # Distinct tokens mapping 1-to-1 to GT words preserve exact token start and end times
        tokens = [
            {
                "word": "ataleniskv",
                "start_time": 1.0,
                "end_time": 1.9,
                "confidence": 0.9,
            },
            {"word": "yisdv", "start_time": 2.0, "end_time": 2.9, "confidence": 0.9},
            {"word": "gesv", "start_time": 3.0, "end_time": 3.9, "confidence": 0.9},
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎨᏒ",
                "raw_phonetic": "A-da-le-ni-s-gv yi-s-dv ge-sv",
                "normalized_text": "ataleniskv yisdv gesv",
                "english": "The beginning of the gospel",
            }
        ]
        result = align_tokens_to_verses(tokens, verses, audio_source="mark_01.wav")
        words = result.verses[0].words
        self.assertEqual(len(words), 3)
        self.assertEqual(words[0].start_sec, 1.0)
        self.assertEqual(words[0].end_sec, 1.9)
        self.assertEqual(words[1].start_sec, 2.0)
        self.assertEqual(words[1].end_sec, 2.9)
        self.assertEqual(words[2].start_sec, 3.0)
        self.assertEqual(words[2].end_sec, 3.9)

    def test_alignment_metrics(self):
        tokens = [
            {
                "word": "ataleniskv",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": 0.95,
            },
            {"word": "yisthv", "start_time": 2.1, "end_time": 3.0, "confidence": 0.90},
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ",
                "raw_phonetic": "A-da-le-ni-s-gv yi-s-dv",
                "normalized_text": "ataleniskv yisthv",
                "english": "The beginning of the gospel",
            }
        ]
        result = align_tokens_to_verses(tokens, verses, audio_source="test.wav")
        self.assertIsNotNone(result.metrics)
        self.assertEqual(result.metrics.matched_verses, 1)
        self.assertLessEqual(result.metrics.overall_cer, 0.10)
        self.assertLessEqual(result.verses[0].cer, 0.10)

    def test_multi_token_asr_fusion_to_gt(self):
        # Two split ASR tokens ("word1wo", "ord2") mapping onto single GT segment ("word1 word2")
        tokens = [
            {"word": "word1wo", "start_time": 1.0, "end_time": 1.5, "confidence": 0.88},
            {"word": "ord2", "start_time": 1.6, "end_time": 2.1, "confidence": 0.85},
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "word1 word2",
                "raw_phonetic": "word1 word2",
                "normalized_text": "word1 word2",
                "english": "test segment",
            }
        ]

        result = align_tokens_to_verses(tokens, verses, audio_source="test.wav")
        v0 = result.verses[0]
        self.assertEqual(len(v0.words), 1)
        self.assertEqual(v0.words[0].word, "word1 word2")
        self.assertEqual(v0.words[0].start_sec, 1.0)
        self.assertEqual(v0.words[0].end_sec, 2.1)

    def test_distinct_words_do_not_fuse(self):
        # Distinct words ("nvhskayohi'a", "tsinikvnha") should align 1-to-1 against GT ("nasgiya hia", "tsinigvnv")
        # rather than being false-fused together into a single segment.
        tokens = [
            {
                "word": "nvhskayohi'a",
                "start_time": 1.0,
                "end_time": 1.5,
                "confidence": 0.90,
            },
            {
                "word": "tsinikvnha",
                "start_time": 1.6,
                "end_time": 2.1,
                "confidence": 0.90,
            },
        ]
        verses = [
            {
                "line_id": "020101",
                "cherokee_syllabary": "ᎾᏍᎩᏯ ᎯᎠ ᏥᏂᎬᏅ",
                "raw_phonetic": "Na-s-gi-ya hi-a tsi-ni-gv-nv",
                "normalized_text": "nasgiya hia tsinigvnv",
                "english": "as it is written",
            }
        ]

        result = align_tokens_to_verses(tokens, verses, audio_source="test.wav")
        v0 = result.verses[0]
        # Should align as 2 distinct word intervals, not fused into 1 interval
        self.assertEqual(len(v0.words), 2)


if __name__ == "__main__":
    unittest.main()
