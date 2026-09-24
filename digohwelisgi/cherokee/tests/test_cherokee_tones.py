# -*- coding: utf-8 -*-
"""
test_tones.py

Unit tests for digohwelisgi.cherokee.orthography.tones.
Tests tone stripping, vowel length/colon normalization, and consonant respelling.
"""

import pytest

from digohwelisgi.cherokee.orthography.tones import (
    remove_tones_and_double_vowels,
    replace_tones,
    respell_consonants,
    strip_tones_and_colons,
)


def test_respell_consonants():
    # t -> th (not ts), d -> t, k -> kh, g -> k, j -> ts, ch -> tsh, hn -> nh
    assert respell_consonants("tsalagi") == "tsalaki"
    assert respell_consonants("gawonihisdi") == "kawonihihsti"
    assert respell_consonants("hna") == "nha"
    assert respell_consonants("hla") == "lha"


def test_strip_tones_and_colons():
    raw = "tsala:ki2 ga:wo2ni:hi3sdi"
    stripped = strip_tones_and_colons(raw)
    assert stripped == "tsalaki gawonihisdi"


def test_replace_tones():
    # Test mapped tones
    text = "a:"
    res, dropped = replace_tones(text)
    assert not dropped
    assert res is not None
    assert "22" in res  # colon maps to 22


def test_remove_tones_and_double_vowels():
    # Colons double vowels
    text = "a:"
    res, dropped = remove_tones_and_double_vowels(text)
    assert not dropped
    assert res is not None
    assert "aa" in res
