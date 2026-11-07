# -*- coding: utf-8 -*-
"""
test_tones_normalization.py

Unit tests for Phase 1 Orthography & Spelling Normalization Engine (Standard 2 TTH).
"""

import pytest

from digohwelisgi.cherokee.orthography import (
    Orthography,
    collapse_double_vowels,
    convert_orthography,
    resolve_vowel_hiatus,
    respell_consonants,
    standardize_glottal_stops,
)


def test_sibilant_word_initial_vs_medial() -> None:
    """
    Verify Standard 2 contextual sibilant pre-aspiration:
    - Word-initial /s/ remains bare s- across all words in a sentence (zero leading hs-).
    - Postvocalic / medial /s/ becomes hs.
    """
    res_pair = collapse_double_vowels(respell_consonants("sakwu sookwili"))
    assert res_pair == "sakwhu sokwhili"
    for word in res_pair.split():
        assert not word.startswith("hs")

    res_cluster = collapse_double_vowels(respell_consonants("sgwaalheehsdi"))
    assert res_cluster == "skwalhehsti"
    assert not res_cluster.startswith("hs")

    # Pre-existing leading hs- is also normalized to bare s-
    res_pre_hs = collapse_double_vowels(respell_consonants("hsgwaalheehsdi"))
    assert res_pre_hs == "skwalhehsti"

    res_medial = respell_consonants("jihsgwa")
    assert res_medial == "tsihskwa"

    # Also verify via convert_orthography(DG -> TTH)
    conv = convert_orthography(
        "sagwu sogwili sgwahlesdi jisgwa",
        source=Orthography.DG,
        target=Orthography.TTH,
    )
    assert conv == "sakwu sokwili skwalhehsti tsihskwa"
    for word in conv.split():
        assert not word.startswith("hs")


def test_glottal_stop_variants() -> None:
    """
    Verify that all glottal stop variants (', ’, ʼ, ʔ, ‚, ?) map to canonical ASCII '
    and illegal post-consonantal glottal stops (*C') are stripped.
    """
    variants = ["'", "’", "ʼ", "ʔ", "‚", "?"]
    for v in variants:
        assert standardize_glottal_stops(v) == "'"
        assert standardize_glottal_stops(f"inake{v}i") == "inake'i"
        assert standardize_glottal_stops(f"ta{v}li") == "ta'li"

    # Strip illegal post-consonantal glottal stops (*C')
    assert standardize_glottal_stops("sagwakt’i") == "sagwakti"
    assert standardize_glottal_stops("kehs'i") == "kehsi"
    assert standardize_glottal_stops("kvyl'i") == "kvyli"

    # Preserve valid word-initial glottal stop before vowel ('a), strip before consonant (*'C)
    assert standardize_glottal_stops("'ama") == "'ama"
    assert standardize_glottal_stops("’yuniyvha") == "yuniyvha"


def test_vowel_hiatus_insertion() -> None:
    """
    Verify epenthetic glottal stop insertion between adjacent vowel nuclei across syllables,
    without splitting plain long doubled vowels within the same syllable nucleus.
    """
    assert resolve_vowel_hiatus("inakei") == "inake'i"
    assert resolve_vowel_hiatus("amai") == "ama'i"
    assert resolve_vowel_hiatus("u:nade:tiyi:sgv̋:i") == "u:nade:tiyi:sgv̋:'i"

    # Plain doubled vowels representing long moras in a single syllable nucleus stay intact
    assert resolve_vowel_hiatus("sookwili") == "sookwili"
    assert resolve_vowel_hiatus("skwaalheehsti") == "skwaalheehsti"


def test_consonant_rewrite_cascades() -> None:
    """
    Verify strict priority ordering of consonant rewrites:
    - d -> t (does NOT cascade to th)
    - t(?!s) -> th
    - g -> k (does NOT cascade to kh)
    - k -> kh
    - j -> ts, ch -> tsh
    - hn -> nh, hl -> lh, hw -> wh, hy -> yh
    - sl before vowels -> slh
    """
    # d -> t without cascading to th; t -> th
    assert respell_consonants("da") == "ta"
    assert respell_consonants("ta") == "tha"
    assert respell_consonants("dada tata") == "tata thatha"

    # g -> k without cascading to kh; k -> kh
    assert respell_consonants("ga") == "ka"
    assert respell_consonants("ka") == "kha"
    assert respell_consonants("gaga kaka") == "kaka khakha"

    # Affricates: j -> ts, ch -> tsh, ts stays ts
    assert respell_consonants("ja") == "tsa"
    assert respell_consonants("cha") == "tsha"
    assert respell_consonants("tsa") == "tsa"

    # Voiceless sonorants: hn -> nh, hl -> lh, hw -> wh, hy -> yh
    assert respell_consonants("hna") == "nha"
    assert respell_consonants("hla") == "lha"
    assert respell_consonants("hwa") == "wha"
    assert respell_consonants("hya") == "yha"

    # Sibilant-lateral cluster before vowel: sl -> slh
    assert respell_consonants("sla") == "slha"
    assert respell_consonants("asla") == "ahslha"
