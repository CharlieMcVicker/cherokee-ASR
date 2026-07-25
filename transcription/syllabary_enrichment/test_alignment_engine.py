# -*- coding: utf-8 -*-
"""
test_alignment_engine.py

Unit tests for character and syllable level alignment engine in transcription.syllabary_enrichment.
"""

import unittest
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
    align_character_syllable_detailed,
    SyllableAlignment,
    get_base_transliteration,
    is_cherokee_syllable,
)


class TestAlignmentEngine(unittest.TestCase):
    def test_is_cherokee_syllable(self):
        self.assertTrue(is_cherokee_syllable("Ꭳ"))
        self.assertTrue(is_cherokee_syllable("Ꮣ"))
        self.assertTrue(is_cherokee_syllable("Ꮝ"))
        self.assertFalse(is_cherokee_syllable("a"))
        self.assertFalse(is_cherokee_syllable(" "))
        self.assertFalse(is_cherokee_syllable("1"))

    def test_get_base_transliteration(self):
        syl = "ᎠᏓᎴᏂᏍᎬ"
        expected = "ataleniskv"
        self.assertEqual(get_base_transliteration(syl), expected)

    def test_align_character_syllable_basic(self):
        syllabary = "ᎠᏓᎴᏂᏍᎬ"
        emitted = "ataleniskv"
        pairs = align_character_syllable(syllabary, emitted)

        self.assertEqual(len(pairs), len(syllabary))
        # Ensure characters match ground truth syllabary ordering
        for (syl_char, _), orig_char in zip(pairs, syllabary):
            self.assertEqual(syl_char, orig_char)

        # Combined emitted output should equal or closely align to emitted string
        reconstructed_emitted = "".join(p[1] for p in pairs)
        self.assertEqual(reconstructed_emitted, emitted)

    def test_align_character_syllable_detailed(self):
        syllabary = "Ꭳ ᎠᏓᎴᏂᏍᎬ"
        emitted = "o ataleniskv"
        alignments = align_character_syllable_detailed(syllabary, emitted)

        self.assertEqual(len(alignments), len(syllabary))
        self.assertIsInstance(alignments[0], SyllableAlignment)
        self.assertEqual(alignments[0].syllabary_char, "Ꭳ")
        self.assertEqual(alignments[0].emitted_text, "o")

        # Check space alignment
        space_align = alignments[1]
        self.assertEqual(space_align.syllabary_char, " ")
        self.assertEqual(space_align.emitted_text, " ")

    def test_align_character_syllable_laryngeal_variation(self):
        # Cherokee "ᏱᏍᏛ" (yistv) aligned with emitted "yisthv" (aspirated /th/)
        syllabary = "ᏱᏍᏛ"
        emitted = "yisthv"
        pairs = align_character_syllable(syllabary, emitted)

        self.assertEqual(len(pairs), len(syllabary))
        # 'Ᏹ' -> 'yi'
        # 'Ꮝ' -> 's'
        # 'Ꮫ' -> 'thv' (laryngeal aspirated variant of tv)
        syl_map = dict(pairs)
        self.assertEqual(syl_map["Ᏹ"], "yi")
        self.assertEqual(syl_map["Ꮝ"], "s")
        self.assertEqual(syl_map["Ꮫ"], "thv")

    def test_align_character_syllable_empty_inputs(self):
        self.assertEqual(align_character_syllable("", "ataleniskv"), [])

        pairs = align_character_syllable("Ꭳ", "")
        self.assertEqual(pairs, [("Ꭳ", "")])


if __name__ == "__main__":
    unittest.main()
