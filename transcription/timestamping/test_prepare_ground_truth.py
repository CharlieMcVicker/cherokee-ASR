# -*- coding: utf-8 -*-
"""
test_prepare_ground_truth.py

Unit tests for prepare_ground_truth.py
"""

import unittest
from transcription.timestamping.prepare_ground_truth import normalize_text_for_alignment


class TestPrepareGroundTruth(unittest.TestCase):
    def test_normalize_text_for_alignment(self):
        raw = "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv, Tsi-sa Ga-lo-ne-dv U-ne-la-nv-hi U-we-tsi u-tse-li-ga."
        normalized = normalize_text_for_alignment(raw)

        # Hyphens removed
        self.assertNotIn("-", normalized)

        # /h/s removed
        self.assertNotIn("h", normalized)

        # Punctuation removed
        self.assertNotIn(",", normalized)
        self.assertNotIn(".", normalized)

        # d -> t, g -> k respelling verified
        self.assertIn("ataleniskv", normalized)

    def test_parse_chunk_list(self):
        import tempfile
        import json
        from transcription.timestamping.prepare_ground_truth import parse_chunk_list

        chunk_data = [
            {
                "line_id": "seg_01",
                "raw_phonetic": "Na-s-gi-ya hi-a",
                "cherokee_syllabary": "ᎾᏍᎩᏯ ᎯᎠ",
                "english": "as it is",
            }
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(chunk_data, f)
            tmp_path = f.name

        try:
            segments = parse_chunk_list(tmp_path)
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["line_id"], "seg_01")
            self.assertEqual(segments[0]["raw_phonetic"], "Na-s-gi-ya hi-a")
            self.assertEqual(segments[0]["normalized_text"], "naskiya ia")
        finally:
            import os

            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
