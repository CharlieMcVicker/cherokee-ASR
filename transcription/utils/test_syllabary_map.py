# -*- coding: utf-8 -*-
"""
test_syllabary_map.py

Unit tests for the centralized Cherokee Syllabary mapping module transcription.utils.syllabary_map.
"""

import unittest
from transcription.utils.syllabary_map import (
    CHEROKEE_SYLLABARY_MAP,
    cherokee_to_bad_phonetics,
)


class TestSyllabaryMap(unittest.TestCase):
    def test_hna_maps_to_nha(self):
        """Ensure Ꮏ is mapped to 'nha' per respell_consonants rules."""
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꮏ"], "nha")

    def test_respell_consonants_applied_consistently(self):
        """Ensure respell_consonants transforms base phonetics as expected."""
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꭰ"], "a")
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꭶ"], "ka")
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꭷ"], "kha")
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꮣ"], "ta")
        self.assertEqual(CHEROKEE_SYLLABARY_MAP["Ꮤ"], "tha")

    def test_cherokee_to_bad_phonetics(self):
        """Ensure transliteration function converts syllabary while preserving punctuation/unknown chars."""
        self.assertEqual(cherokee_to_bad_phonetics("ᎣᏏᏲ"), "osiyo")
        self.assertEqual(cherokee_to_bad_phonetics("Ꮏ!"), "nha!")


if __name__ == "__main__":
    unittest.main()
