import unittest
from transcription.utils.syllabary_map import (
    CHEROKEE_SYLLABARY_MAP,
    cherokee_to_bad_phonetics,
    phonetics_to_syllabary,
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
        """Ensure transliteration function converts syllabary while preserving punctuation/unknown chars and inserting hiatus glottals."""
        self.assertEqual(cherokee_to_bad_phonetics("ᎣᏏᏲ"), "ohsiyo")
        self.assertEqual(cherokee_to_bad_phonetics("Ꮏ!"), "nha!")
        self.assertEqual(cherokee_to_bad_phonetics("ᎢᎾᎨᎢ"), "inake'i")
        self.assertEqual(cherokee_to_bad_phonetics("ᎯᎠ"), "hi'a")
        self.assertEqual(cherokee_to_bad_phonetics("ᎠᏍᎦᏅᏨᎢ"), "ahskanvtsv'i")

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
