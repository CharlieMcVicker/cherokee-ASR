# -*- coding: utf-8 -*-
"""
test_orthography.py

Unit tests for transcription.cherokee.orthography.
Tests Orthography enum, convert_orthography, string cleaners, and tone decoupling.
"""

import pytest

from transcription.cherokee.orthography import (
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    strip_tones_and_colons,
)


def test_convert_orthography_short_circuit():
    text = "na olha akhvhskwohstohti ehskhvhsi"
    res = convert_orthography(text, source=Orthography.TTH, target=Orthography.TTH)
    assert res == text
    assert "hh" not in res
    assert "khh" not in res
    assert "thh" not in res


def test_dg_to_tth_conversion():
    text = "na otla akvskwostoti eskvsi"
    converted = convert_orthography(text, source=Orthography.DG, target=Orthography.TTH)
    assert "akh" in converted
    assert "hskw" in converted
    assert "hh" not in converted


def test_syllabary_to_tth_conversion():
    syl = "Ꮎ ᎣᏝ ᎠᎬᏍᏉᏍᏙᏗ ᎡᏍᎬᏏ"
    converted = convert_orthography(
        syl, source=Orthography.SYLLABARY, target=Orthography.TTH
    )
    assert converted == "na otla akvhskwohstoti ehskvhsi"
    assert "hh" not in converted


def test_syllabary_to_tth_contextual_vs_unconditional():
    text = "ᏍᎩ ᏌᏊ ᎠᏍᎦᏯ ᏣᎳᎩ"
    contextual = convert_orthography(
        text,
        source=Orthography.SYLLABARY,
        target=Orthography.TTH,
        contextual_preaspiration=True,
    )
    assert contextual == "ski sakwu ahskaya tsalaki"

    unconditional = convert_orthography(
        text,
        source=Orthography.SYLLABARY,
        target=Orthography.TTH,
        contextual_preaspiration=False,
    )
    assert unconditional == "hski hsakwu ahskaya tsalaki"


def test_syllabary_to_dg_conversion():
    syl = "ᎢᎾᎨᎢ"
    converted = convert_orthography(
        syl, source=Orthography.SYLLABARY, target=Orthography.DG
    )
    assert converted == "inake'i"


def test_strip_tones_and_colons():
    toned_text = "tsala:ki2 ga:wo2ni:hi3sdi"
    assert strip_tones_and_colons(toned_text) == "tsalaki gawonihisdi"

    converted = convert_orthography(
        toned_text, source=Orthography.DG, target=Orthography.TTH, strip_tones=True
    )
    assert "2" not in converted
    assert "3" not in converted
    assert ":" not in converted


def test_clean_punctuation_and_whitespace():
    raw = "  ᎢᎾᎨᎢ...  , ?  `tsi-sa` !  "
    cleaned = clean_punctuation_and_whitespace(raw)
    assert "..." not in cleaned
    assert "," not in cleaned
    assert "?" not in cleaned
    assert "'" in cleaned  # ` converted to '


def test_empty_string():
    assert convert_orthography("", Orthography.DG, Orthography.TTH) == ""
    assert clean_punctuation_and_whitespace("") == ""
    assert strip_tones_and_colons("") == ""
