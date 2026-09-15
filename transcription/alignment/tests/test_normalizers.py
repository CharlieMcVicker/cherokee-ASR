# -*- coding: utf-8 -*-
"""
Unit tests for transcription.alignment.normalizers.
"""

import pytest
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
    normalize_text_for_alignment,
)


def test_normalize_syllabary_for_alignment():
    # Syllabary transliteration with hyphens, case, punctuation
    result = normalize_syllabary_for_alignment("A-da-le-ni-s-gv.")
    assert result == "atalenihskv"
    assert "-" not in result
    assert "." not in result


def test_normalize_syllabary_hiatus_glottal_stops():
    # Syllabary with adjacent vowels receives hiatus glottal stop
    assert normalize_syllabary_for_alignment("ᎢᎾᎨᎢ") == "inake'i"
    assert normalize_syllabary_for_alignment("ᎯᎠ") == "hi'a"
    assert normalize_syllabary_for_alignment("ᎠᏍᎦᏅᏨᎢ") == "askanvtsv'i"
    assert normalize_syllabary_for_alignment("ᎣᏏᏲ") == "osiyo"


def test_normalize_phonetics_for_alignment_preserves_h():
    # Word-initial 'h' is preserved
    assert normalize_phonetics_for_alignment("ho-wa") == "howa"
    assert normalize_phonetics_for_alignment("hi-la") == "hila"

    # Phonetic text preserves aspiration 'h'
    result = normalize_phonetics_for_alignment("A-da-le-ni-s-gv.")
    assert result == "atalenihskv"
    assert "h" in result
    assert "-" not in result
    assert "." not in result


def test_normalizers_qu_conversion():
    # 'qu' converted to 'gw' and respelled
    syll_res = normalize_syllabary_for_alignment("quana")
    phon_res = normalize_phonetics_for_alignment("quana")
    assert "qu" not in syll_res
    assert "qu" not in phon_res


def test_normalizers_empty_and_whitespace():
    assert normalize_syllabary_for_alignment("") == ""
    assert normalize_syllabary_for_alignment("   ") == ""
    assert normalize_phonetics_for_alignment("") == ""
    assert normalize_phonetics_for_alignment("   ") == ""


def test_normalize_text_for_alignment_backwards_compat():
    # normalize_text_for_alignment is an alias for normalize_syllabary_for_alignment
    assert normalize_text_for_alignment(
        "A-da-le-ni-s-gv."
    ) == normalize_syllabary_for_alignment("A-da-le-ni-s-gv.")
    assert normalize_text_for_alignment("ho-wa") == "howa"
