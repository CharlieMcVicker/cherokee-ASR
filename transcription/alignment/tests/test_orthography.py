# -*- coding: utf-8 -*-
"""
Unit test suite for Orthography enum and explicit conversion maps.
"""

import pytest

from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.utils.orthography import (
    Orthography,
    convert_orthography,
    strip_tones_and_colons,
)


def test_convert_orthography_short_circuit_same_source_and_target():
    text = "na olha akhvhskwohstohti ehskhvhsi"
    # When source == target, must return exact clean string without any consonant mutation
    res = convert_orthography(text, source=Orthography.TTH, target=Orthography.TTH)
    assert res == text
    assert "hh" not in res
    assert "khh" not in res
    assert "thh" not in res


def test_dg_to_tth_conversion():
    # Base d/g input converting to aspirated t/th
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


def test_syllabary_to_dg_conversion():
    syl = "ᎢᎾᎨᎢ"
    converted = convert_orthography(
        syl, source=Orthography.SYLLABARY, target=Orthography.DG
    )
    assert converted == "inake'i"


def test_tone_orthogonality():
    toned_text = "tsala:ki2 ga:wo2ni:hi3sdi"
    # Tone stripping is decoupled from consonant systems
    assert strip_tones_and_colons(toned_text) == "tsalaki gawonihisdi"

    converted = convert_orthography(
        toned_text, source=Orthography.DG, target=Orthography.TTH, strip_tones=True
    )
    assert "2" not in converted
    assert "3" not in converted
    assert ":" not in converted


def test_idempotency_of_normalize_phonetics_for_alignment():
    samples = [
        "ehskhvhsi",
        "na olha akhvhskwohstohti ehskhvhsi",
        "thothalv khalvla'thi aneho awoha'li",
        "khakho vhskhina akheyha",
        "tsana'ahs thakwhalela",
    ]

    for s in samples:
        norm1 = normalize_phonetics_for_alignment(s, source=Orthography.TTH)
        norm2 = normalize_phonetics_for_alignment(norm1, source=Orthography.TTH)
        norm3 = normalize_phonetics_for_alignment(norm2, source=Orthography.TTH)

        assert norm1 == norm2 == norm3
        assert "hh" not in norm1
        assert "thh" not in norm1
        assert "khh" not in norm1


def test_empty_string_handling():
    assert convert_orthography("", Orthography.DG, Orthography.TTH) == ""
    assert normalize_phonetics_for_alignment("") == ""
    assert normalize_syllabary_for_alignment("") == ""
