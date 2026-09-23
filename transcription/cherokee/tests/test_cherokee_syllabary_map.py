# -*- coding: utf-8 -*-
"""
test_syllabary_map.py

Unit tests for transcription.cherokee.orthography.syllabary_map.
Tests Cherokee Unicode syllabary lookup dictionaries and bidirectional conversions.
"""

import pytest

from transcription.cherokee.orthography.syllabary_map import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    phonetics_to_syllabary,
    syllabary_to_phonetics,
)


def test_hna_maps_to_nha():
    assert CHEROKEE_SYLLABARY_MAP["Ꮏ"] == "nha"
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꮏ"] == "nha"


def test_base_syllabary_mappings():
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꭰ"] == "a"
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꭶ"] == "ka"
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꭷ"] == "kha"
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꮣ"] == "ta"
    assert CHEROKEE_SYLLABARY_BASE_MAP["Ꮤ"] == "tha"


def test_respell_consonants_applied_consistently():
    assert CHEROKEE_SYLLABARY_MAP["Ꭰ"] == "a"
    assert CHEROKEE_SYLLABARY_MAP["Ꭶ"] == "ka"
    assert CHEROKEE_SYLLABARY_MAP["Ꭷ"] == "kha"
    assert CHEROKEE_SYLLABARY_MAP["Ꮣ"] == "ta"
    assert CHEROKEE_SYLLABARY_MAP["Ꮤ"] == "tha"


def test_unconditional_sibilants():
    assert CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP["Ꮜ"] == "hsa"
    assert CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP["Ꮝ"] == "hs"
    assert CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP["Ꮞ"] == "hse"


def test_syllabary_to_phonetics():
    assert syllabary_to_phonetics("ᎣᏏᏲ") == "ohsiyo"
    assert syllabary_to_phonetics("Ꮏ!") == "nha!"
    assert syllabary_to_phonetics("ᎢᎾᎨᎢ") == "inake'i"
    assert syllabary_to_phonetics("ᎯᎠ") == "hi'a"
    assert syllabary_to_phonetics("ᎠᏍᎦᏅᏨᎢ") == "ahskanvtsv'i"


def test_syllabary_to_phonetics_contextual_vs_unconditional():
    assert syllabary_to_phonetics("ᏍᎩ", contextual_preaspiration=True) == "ski"
    assert syllabary_to_phonetics("ᏍᎩ", contextual_preaspiration=False) == "hski"
    assert syllabary_to_phonetics("ᏌᏊ", contextual_preaspiration=True) == "sakwu"
    assert syllabary_to_phonetics("ᏌᏊ", contextual_preaspiration=False) == "hsakwu"
    assert syllabary_to_phonetics("ᎠᏍᎦᏯ", contextual_preaspiration=True) == "ahskaya"
    assert syllabary_to_phonetics("ᎠᏍᎦᏯ", contextual_preaspiration=False) == "ahskaya"
    assert syllabary_to_phonetics("ᎣᏏᏲ", contextual_preaspiration=True) == "ohsiyo"
    assert syllabary_to_phonetics("ᎣᏏᏲ", contextual_preaspiration=False) == "ohsiyo"
    assert syllabary_to_phonetics("ᏣᎳᎩ", contextual_preaspiration=True) == "tsalaki"
    assert syllabary_to_phonetics("ᏣᎳᎩ", contextual_preaspiration=False) == "tsalaki"
    assert syllabary_to_phonetics("ᏥᏍᏆ", contextual_preaspiration=True) == "tsihskwa"
    assert syllabary_to_phonetics("ᏥᏍᏆ", contextual_preaspiration=False) == "tsihskwa"


def test_phonetics_to_syllabary():
    assert phonetics_to_syllabary("ka") == "Ꭶ"
    assert phonetics_to_syllabary("kha") == "Ꭷ"
    assert phonetics_to_syllabary("ta") == "Ꮣ"
    assert phonetics_to_syllabary("tha") == "Ꮤ"
    assert phonetics_to_syllabary("ohsiyo") == "ᎣᏏᏲ"
    assert phonetics_to_syllabary("kanolv'vhska") == "ᎦᏃᎸᎥᏍᎦ"
    assert phonetics_to_syllabary("kakhahiya") == "ᎦᎧᎯᏯ"
    assert phonetics_to_syllabary("thv") == "Ꮫ"  # h-drop fallback
    assert phonetics_to_syllabary("khv") == "Ꭼ"  # h-drop fallback
    assert phonetics_to_syllabary("kakhahv'a") == "ᎦᎧᎲᎠ"
