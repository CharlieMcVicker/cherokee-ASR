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


if __name__ == "__main__":
    unittest.main()
