import unittest
from transcription.utils.syllabary_map import (
    CHEROKEE_SYLLABARY_MAP,
    phonetics_to_syllabary,
    syllabary_to_phonetics,
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

    def test_syllabary_to_phonetics(self):
        """Ensure transliteration function converts syllabary while preserving punctuation/unknown chars and inserting hiatus glottals."""
        self.assertEqual(syllabary_to_phonetics("ᎣᏏᏲ"), "ohsiyo")
        self.assertEqual(syllabary_to_phonetics("Ꮏ!"), "nha!")
        self.assertEqual(syllabary_to_phonetics("ᎢᎾᎨᎢ"), "inake'i")
        self.assertEqual(syllabary_to_phonetics("ᎯᎠ"), "hi'a")
        self.assertEqual(syllabary_to_phonetics("ᎠᏍᎦᏅᏨᎢ"), "ahskanvtsv'i")

    def test_syllabary_to_phonetics_contextual_vs_unconditional(self):
        """Ensure contextual preaspiration suppresses leading 'h' on word-initial s and affricates, while unconditional preserves hs everywhere."""
        # Word-initial 's' glyph (Ꮝ)
        self.assertEqual(
            syllabary_to_phonetics("ᏍᎩ", contextual_preaspiration=True), "ski"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᏍᎩ", contextual_preaspiration=False), "hski"
        )

        # Word-initial 'sa' glyph (Ꮜ)
        self.assertEqual(
            syllabary_to_phonetics("ᏌᏊ", contextual_preaspiration=True), "sakwu"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᏌᏊ", contextual_preaspiration=False), "hsakwu"
        )

        # Medial postvocalic 's' glyph (Ꮝ)
        self.assertEqual(
            syllabary_to_phonetics("ᎠᏍᎦᏯ", contextual_preaspiration=True), "ahskaya"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᎠᏍᎦᏯ", contextual_preaspiration=False), "ahskaya"
        )

        # Medial postvocalic 'si' glyph (Ꮟ)
        self.assertEqual(
            syllabary_to_phonetics("ᎣᏏᏲ", contextual_preaspiration=True), "ohsiyo"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᎣᏏᏲ", contextual_preaspiration=False), "ohsiyo"
        )

        # Affricate 'ts' series (Ꮳ, Ꮵ) with medial sibilant
        self.assertEqual(
            syllabary_to_phonetics("ᏣᎳᎩ", contextual_preaspiration=True), "tsalaki"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᏣᎳᎩ", contextual_preaspiration=False), "tsalaki"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᏥᏍᏆ", contextual_preaspiration=True), "tsihskwa"
        )
        self.assertEqual(
            syllabary_to_phonetics("ᏥᏍᏆ", contextual_preaspiration=False), "tsihskwa"
        )

    def test_phonetics_to_syllabary_direct_matches(self):
        """Test phonetic transliteration to Cherokee syllabary conversion."""
        self.assertEqual(phonetics_to_syllabary("ka"), "Ꭶ")
        self.assertEqual(phonetics_to_syllabary("kha"), "Ꭷ")
        self.assertEqual(phonetics_to_syllabary("ta"), "Ꮣ")
        self.assertEqual(phonetics_to_syllabary("tha"), "Ꮤ")
        self.assertEqual(phonetics_to_syllabary("ohsiyo"), "ᎣᏏᏲ")
        self.assertEqual(phonetics_to_syllabary("kanolv'vhska"), "ᎦᏃᎸᎥᏍᎦ")
        self.assertEqual(phonetics_to_syllabary("kakhahiya"), "ᎦᎧᎯᏯ")

    def test_phonetics_to_syllabary_h_drop_fallback(self):
        """Test fallback behavior where 'h' is dropped when aspirated token is not found directly."""
        # 'thv' falls back to 'tv' -> Ꮫ
        self.assertEqual(phonetics_to_syllabary("thv"), "Ꮫ")
        # 'khv' falls back to 'kv' -> Ꭼ
        self.assertEqual(phonetics_to_syllabary("khv"), "Ꭼ")

    def test_phonetics_to_syllabary_compound_words(self):
        """Test multi-syllable word and phrase transliteration."""
        self.assertEqual(phonetics_to_syllabary("kakhahv'a"), "ᎦᎧᎲᎠ")


if __name__ == "__main__":
    unittest.main()
