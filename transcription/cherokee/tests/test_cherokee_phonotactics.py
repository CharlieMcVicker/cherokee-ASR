# -*- coding: utf-8 -*-
"""
test_cherokee_phonotactics.py

Unit tests for Cherokee phonotactic rules, tokenizer, syncope masking,
intrusive site masking, and prepare_cherokee_text in transcription.cherokee.phonotactics.
"""

from typing import Any
import numpy as np
import pytest

from transcription.cherokee.phonotactics import (
    ASPIRATED_STOP_SET,
    LARYNGEAL_SET,
    PLAIN_SONORANT_SET,
    PLAIN_STOP_SET,
    SIBILANT_CLUSTER_SET,
    SIBILANT_SET,
    VOICELESS_SONORANT_SET,
    VOWEL_SET,
    PhonemeCategory,
    PhonotacticAnalysis,
    PhonotacticToken,
    analyze_phonotactics,
    get_intrusion_site_mask,
    get_syncope_mask,
    is_valid_phonotactic_sequence,
    prepare_cherokee_text,
    tokenize_phonemes,
)
from transcription.core.alignment.ctc import TextPreparerProtocol


def test_text_preparer_protocol_conformance():
    """Verify prepare_cherokee_text implements TextPreparerProtocol."""
    assert isinstance(prepare_cherokee_text, TextPreparerProtocol)


def test_tokenize_all_canonical_digraphs_and_trigraphs():
    """Verify tokenizer correctly segments and categorizes canonical multi-character Cherokee tokens."""
    sample = "kwh tlh tsh th kh kw ts tl hs lh nh wh yh hsk hst hskw hsl hsts slh"
    tokens = tokenize_phonemes(sample)

    expected_symbols = [
        ("kwh", PhonemeCategory.ASPIRATED_STOP),
        ("tlh", PhonemeCategory.ASPIRATED_STOP),
        ("tsh", PhonemeCategory.ASPIRATED_STOP),
        ("th", PhonemeCategory.ASPIRATED_STOP),
        ("kh", PhonemeCategory.ASPIRATED_STOP),
        ("kw", PhonemeCategory.PLAIN_STOP),
        ("ts", PhonemeCategory.PLAIN_STOP),
        ("tl", PhonemeCategory.PLAIN_STOP),
        ("hs", PhonemeCategory.SIBILANT),
        ("lh", PhonemeCategory.VOICELESS_SONORANT),
        ("nh", PhonemeCategory.VOICELESS_SONORANT),
        ("wh", PhonemeCategory.VOICELESS_SONORANT),
        ("yh", PhonemeCategory.VOICELESS_SONORANT),
        ("hsk", PhonemeCategory.SIBILANT_CLUSTER),
        ("hst", PhonemeCategory.SIBILANT_CLUSTER),
        ("hskw", PhonemeCategory.SIBILANT_CLUSTER),
        ("hsl", PhonemeCategory.SIBILANT_CLUSTER),
        ("hsts", PhonemeCategory.SIBILANT_CLUSTER),
        ("slh", PhonemeCategory.SIBILANT_CLUSTER),
    ]

    non_space_tokens = [t for t in tokens if t.category != PhonemeCategory.OTHER]
    assert len(non_space_tokens) == len(expected_symbols)
    for tok, (sym, cat) in zip(non_space_tokens, expected_symbols):
        assert tok.symbol == sym
        assert tok.category == cat


def test_syncope_mask_generation():
    """Verify syncope masking correctly marks eligible short vowels."""
    # In 'ataleniskv', vowels 'a', 'e', 'i' after consonants are candidates for syncope
    text = "ataleniskv"
    tokens = tokenize_phonemes(text)
    syncope_mask = get_syncope_mask(tokens, return_char_mask=False)
    assert len(syncope_mask) == len(tokens)

    # Initial vowel 'a' should not syncope (no preceding consonant)
    assert syncope_mask[0] is False


def test_lateral_deaffrication_syncope():
    """Verify 'tl' and 'tlh' initial stop occlusion can be marked for syncope."""
    text = "otla"
    char_mask = get_syncope_mask(text, return_char_mask=True)
    # 't' in 'tl' is at index 1
    assert char_mask[1] is True


def test_intrusion_site_mask():
    """Verify intrusion site masking for laryngeals."""
    text = "osiyo"
    char_mask = get_intrusion_site_mask(text, return_char_mask=True)
    assert len(char_mask) == len(text)
    # Initial vowel is intrusion site for initial onset
    assert char_mask[0] is True


def test_phonotactic_constraints():
    """Verify *HH, *C', and *ChR constraints."""
    assert is_valid_phonotactic_sequence("osiyo") is True
    assert is_valid_phonotactic_sequence("athaleniskv") is True

    # *HH: adjacent laryngeals
    assert is_valid_phonotactic_sequence("a'h") is False
    assert is_valid_phonotactic_sequence("ah'a") is False
    assert is_valid_phonotactic_sequence("a'hs") is False

    # *C': post-consonantal glottal stop
    assert is_valid_phonotactic_sequence("t'a") is False
    assert is_valid_phonotactic_sequence("k'a") is False

    # *ChR: stop + voiceless sonorant
    assert is_valid_phonotactic_sequence("thlh") is False
    assert is_valid_phonotactic_sequence("khnh") is False


def test_analyze_phonotactics():
    """Verify full phonotactic analysis container."""
    text = "hi'a"
    res = analyze_phonotactics(text)
    assert isinstance(res, PhonotacticAnalysis)
    assert res.text == text
    assert len(res.char_syncope_mask) == len(text)
    assert len(res.char_intrusion_mask) == len(text)


def test_prepare_cherokee_text_conformance():
    """Verify prepare_cherokee_text produces expected ground truth matrix and masks."""

    class DummyConfig:
        space = " "
        blank = 0
        is_syncope_token: Any = None
        is_intrusive_site: Any = None

    config = DummyConfig()
    char_list = ["<blank>", " ", "a", "t", "l", "i", "s", "k", "v"]
    gt_mat, utt_begin = prepare_cherokee_text(
        config,
        ["atli", "skv"],
        char_list=char_list,
        enforce_phonotactics=True,
    )

    assert isinstance(gt_mat, np.ndarray)
    assert gt_mat.ndim == 2
    assert len(utt_begin) == 3
    assert config.is_syncope_token is not None
    assert config.is_intrusive_site is not None
    assert len(config.is_syncope_token) == len(gt_mat)
