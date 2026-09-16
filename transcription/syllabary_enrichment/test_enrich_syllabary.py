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
        base_trans = get_base_transliteration(syllabary)  # "atalenihskv"

        # Suppose ASR emitted 'atalenihsk' where final vowel 'v' is dropped (syncopated)
        aligned_pairs = [
            ("Ꭰ", "a"),
            ("Ꮣ", "ta"),
            ("Ꮄ", "le"),
            ("Ꮒ", "ni"),
            ("Ꮝ", "hs"),
            ("Ꭼ", "k"),  # ASR emitted 'k' without vowel 'v'
        ]

        enriched = reconcile_phonetics(
            syllabary, base_trans, "atalenihsk", aligned_pairs
        )
        self.assertEqual(enriched, "atalenihsk")

    def test_reconcile_phonetics_glottals_and_preaspiration(self):
        """Rule 2: Pre-aspiration 'h' and glottal stop ''' transfer from ASR."""
        syllabary = "Ꭰ Ꮣ"
        base_trans = "a ta"

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
        base_trans = "ta ka"

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
        base_trans = "yihstv"  # base transliteration

        # ASR emits 'yihsth' (laryngeal toggle t->th and vowel syncopation v dropped)
        aligned_pairs = align_character_syllable(syllabary, "yihsth")

        enriched = reconcile_phonetics(syllabary, base_trans, "yihsth", aligned_pairs)
        self.assertEqual(enriched, "yihsth")

    def test_reconcile_post_vocalic_aspiration_h(self):
        """Preserve post-vocalic or vocalic aspiration 'h' emitted by ASR."""
        syllabary = "Ᏺ Ꭽ Ꮒ"
        base_trans = "yo ha ni"

        aligned_pairs = [
            ("Ᏺ", "yoh"),
            (" ", " "),
            ("Ꭽ", "hah"),
            (" ", " "),
            ("Ꮒ", "nih"),
        ]

        enriched = reconcile_phonetics(
            syllabary, base_trans, "yoh hah nih", aligned_pairs
        )
        self.assertEqual(enriched, "yoh hah nih")

    def test_reconcile_preaspiration_s_clusters(self):
        """Group pre-aspiration 'h' with s-clusters and syllable boundaries."""
        syllabary = "Ꮝ Ꮣ"
        base_trans = "s ta"

        aligned_pairs = [
            ("Ꮝ", "hs"),
            (" ", " "),
            ("Ꮣ", "ta"),
        ]

        enriched = reconcile_phonetics(syllabary, base_trans, "hs ta", aligned_pairs)
        self.assertEqual(enriched, "hs ta")

    def test_reconcile_tla_to_lha_phonological_shift(self):
        """Phonological toggle tla/tle/tli/tlo/tlu/tlv -> lha/lhe/lhi/lho/lhu/lhv when aligned ASR window contains 'lh'."""
        syllabary = "Ꮭ Ꮮ Ꮯ Ꮰ Ꮱ Ꮲ"
        base_trans = "tla tle tli tlo tlu tlv"

        aligned_pairs = [
            ("Ꮭ", "lhah"),
            (" ", " "),
            ("Ꮮ", "lhe"),
            (" ", " "),
            ("Ꮯ", "lhi"),
            (" ", " "),
            ("Ꮰ", "lho"),
            (" ", " "),
            ("Ꮱ", "lhuh"),
            (" ", " "),
            ("Ꮲ", "lhv"),
        ]

        enriched = reconcile_phonetics(
            syllabary, base_trans, "lhah lhe lhi lho lhuh lhv", aligned_pairs
        )
        self.assertEqual(enriched, "lhah lhe lhi lho lhuh lhv")

    def test_reconcile_l_series_to_lh_lateral_aspiration(self):
        """Liquid l-series (la/le/li/lo/lu/lv) -> lh-series when ASR window contains 'lh', including syncopated vowels."""
        syllabary = "Ꮅ"
        base_trans = "li"
        aligned_pairs = [("Ꮅ", "lh")]

        enriched = reconcile_phonetics(syllabary, base_trans, "lh", aligned_pairs)
        self.assertEqual(enriched, "lh")

    def test_reconcile_aspiration_transfer_with_syncopation(self):
        """As per core rules: aspiration transfer + vowel syncopation (e.g. ka + ASR 'kh' -> 'kh')."""
        syllabary = "Ꭶ"
        base_trans = "ka"
        aligned_pairs = [("Ꭶ", "kh")]

        enriched = reconcile_phonetics(syllabary, base_trans, "kh", aligned_pairs)
        self.assertEqual(enriched, "kh")

    def test_reconcile_drop_standalone_onset_glottal(self):
        """Drop standalone onset glottal stop from ASR unless accompanied by an explicit onset consonant."""
        syllabary = "Ꮿ"
        base_trans = "ya"
        aligned_pairs = [("Ꮿ", "'a")]

        enriched = reconcile_phonetics(syllabary, base_trans, "'a", aligned_pairs)
        self.assertEqual(enriched, "ya")

    def test_reconcile_empty_inputs(self):
        self.assertEqual(reconcile_phonetics("", "", "", []), "")
        self.assertEqual(reconcile_phonetics("Ꭳ", "o", "", [("Ꭳ", "")]), "o")


if __name__ == "__main__":
    unittest.main()
