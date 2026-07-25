# -*- coding: utf-8 -*-
"""
test_enrich_syllabary.py

Unit tests for Phonetic Rule Merger Engine in transcription.syllabary_enrichment.enrich_syllabary.
"""

import unittest
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
    get_base_transliteration,
)
from transcription.syllabary_enrichment.enrich_syllabary import reconcile_phonetics


class TestEnrichSyllabary(unittest.TestCase):
    def test_reconcile_phonetics_syncopation(self):
        """Rule 1: Vowel deletion / syncopation based on aligned ASR window."""
        syllabary = "ᎠᏓᎴᏂᏍᎬ"
        base_trans = get_base_transliteration(syllabary)  # "adalenisgv"

        # Suppose ASR emitted 'atalenisk' where final vowel 'v' is dropped (syncopated)
        aligned_pairs = [
            ("Ꭰ", "a"),
            ("Ꮣ", "ta"),
            ("Ꮄ", "le"),
            ("Ꮒ", "ni"),
            ("Ꮝ", "s"),
            ("Ꭼ", "k"),  # ASR emitted 'k' without vowel 'v'
        ]

        enriched = reconcile_phonetics(
            syllabary, base_trans, "atalenisk", aligned_pairs
        )
        self.assertEqual(enriched, "atalenisk")

    def test_reconcile_phonetics_glottals_and_preaspiration(self):
        """Rule 2: Pre-aspiration 'h' and glottal stop ''' transfer from ASR."""
        syllabary = "Ꭰ Ꮣ"
        base_trans = "a da"

        # ASR emits pre-aspiration and glottal stop: "a' htha"
        aligned_pairs = [
            ("Ꭰ", "a'"),
            (" ", " "),
            ("Ꮣ", "htha"),
        ]

        enriched = reconcile_phonetics(syllabary, base_trans, "a' htha", aligned_pairs)
        self.assertEqual(enriched, "a' htha")

    def test_reconcile_phonetics_laryngeal_toggles(self):
        """Rule 2: Laryngeal toggles t -> th and k -> kh."""
        syllabary = "Ꮣ Ꭶ"
        base_trans = "da ga"

        aligned_pairs = [
            ("Ꮣ", "tha"),
            (" ", " "),
            ("Ꭶ", "kha"),
        ]

        enriched = reconcile_phonetics(syllabary, base_trans, "tha kha", aligned_pairs)
        self.assertEqual(enriched, "tha kha")

    def test_reconcile_phonetics_combined_rules(self):
        """Combined test covering syncopation, glottal injection, and laryngeal toggles."""
        syllabary = "ᏱᏍᏛ"
        base_trans = "yisdv"  # base transliteration

        # ASR emits 'yisth' (laryngeal toggle d->th and vowel syncopation v dropped)
        aligned_pairs = align_character_syllable(syllabary, "yisth")

        enriched = reconcile_phonetics(syllabary, base_trans, "yisth", aligned_pairs)
        self.assertEqual(enriched, "yisth")

    def test_reconcile_empty_inputs(self):
        self.assertEqual(reconcile_phonetics("", "", "", []), "")
        self.assertEqual(reconcile_phonetics("Ꭳ", "o", "", [("Ꭳ", "")]), "o")


if __name__ == "__main__":
    unittest.main()
