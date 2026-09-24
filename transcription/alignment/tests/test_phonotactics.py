# -*- coding: utf-8 -*-
"""
test_phonotactics.py

Unit test suite for Cherokee phonotactic rules, tokenizer, syncope masking,
and intrusive site masking in transcription/alignment/phonotactics.py.
"""

import pytest
from transcription.alignment.phonotactics import (
    PhonemeCategory,
    PhonotacticToken,
    PhonotacticAnalysis,
    tokenize_phonemes,
    get_syncope_mask,
    get_intrusion_site_mask,
    is_valid_phonotactic_sequence,
    analyze_phonotactics,
)


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

    non_space_tokens = [t for t in tokens if t.symbol != " "]
    assert len(non_space_tokens) == len(expected_symbols)

    for tok, (expected_sym, expected_cat) in zip(non_space_tokens, expected_symbols):
        assert tok.symbol == expected_sym
        assert tok.category == expected_cat


def test_tokenize_complex_cherokee_words():
    """Verify tokenizer on real-world Cherokee words from training corpus."""
    word = "ekwhami"
    tokens = tokenize_phonemes(word)
    symbols = [t.symbol for t in tokens]
    categories = [t.category for t in tokens]

    assert symbols == ["e", "kwh", "a", "m", "i"]
    assert categories == [
        PhonemeCategory.VOWEL,
        PhonemeCategory.ASPIRATED_STOP,
        PhonemeCategory.VOWEL,
        PhonemeCategory.PLAIN_SONORANT,
        PhonemeCategory.VOWEL,
    ]


def test_syncope_mask_generation():
    """Verify syncope mask correctly marks vowels in open syllables with consonant onsets."""
    word = "adalenisgv"
    char_mask = get_syncope_mask(word, return_char_mask=True)
    token_mask = get_syncope_mask(word, return_char_mask=False)

    tokens = tokenize_phonemes(word)
    assert len(char_mask) == len(word)
    assert len(token_mask) == len(tokens)

    # In 'adalenisgv' (a-da-le-ni-s-gv):
    # 'a' (initial vowel, no onset consonant -> False)
    # 'a' in 'da' (preceded by 'd'/'t' -> True)
    # 'e' in 'le' (preceded by 'l' -> True)
    # 'i' in 'ni' (preceded by 'n' -> True)
    # 'v' in 'gv' (preceded by 'g'/'k' -> True)
    assert char_mask[0] is False  # 'a'
    assert char_mask[2] is True  # 'a' in 'da'
    assert char_mask[4] is True  # 'e' in 'le'
    assert char_mask[6] is True  # 'i' in 'ni'
    assert char_mask[9] is True  # 'v' in 'gv'


def test_syncope_blocks_forbidden_clusters():
    """Verify syncope is blocked when it would create a forbidden *ChR (stop + voiceless sonorant) cluster."""
    word = "talanha"  # 't' + 'a' + 'l' + 'a' + 'nh' + 'a'
    # Suppose we have 't' + 'a' + 'nh' + 'a' (stop 't' before voiceless sonorant 'nh')
    test_seq = "tanh"
    tokens = tokenize_phonemes(test_seq)
    token_mask = get_syncope_mask(test_seq, return_char_mask=False)

    # 'a' in 'tanh' is followed by 'nh' (voiceless sonorant); deleting 'a' produces *tnh (*ChR)
    # Syncope should be blocked on 'a'
    assert token_mask[1] is False


def test_intrusion_site_mask():
    """Verify intrusion site mask identifies candidate sites for intrusive /h/ and /'/."""
    # In 'atvhska':
    # 't' after 'a' is a candidate intrusion site
    # 'v' is vowel
    # 'hsk' is sibilant cluster (already pre-aspirated)
    word = "atvhska"
    char_mask = get_intrusion_site_mask(word, return_char_mask=True)
    tokens = tokenize_phonemes(word)
    token_mask = get_intrusion_site_mask(word, return_char_mask=False)

    assert len(char_mask) == len(word)
    assert len(token_mask) == len(tokens)

    # 't' (plain stop) at token index 1 is an eligible intrusion site
    t_tok = tokens[1]
    assert t_tok.symbol == "t"
    assert token_mask[1] is True


def test_phonotactic_constraints_validation():
    """Verify surface phonotactic constraints: *HH, *C', and *ChR."""
    # Valid sequences
    assert is_valid_phonotactic_sequence("adalenisgv") is True
    assert is_valid_phonotactic_sequence("ekwhami") is True
    assert is_valid_phonotactic_sequence("nahskhi") is True
    assert is_valid_phonotactic_sequence("tsihsa") is True

    # Violations:
    # 1. *HH constraint (adjacent laryngeals)
    assert is_valid_phonotactic_sequence("a'hsk") is False
    assert is_valid_phonotactic_sequence("ah'sk") is False
    assert is_valid_phonotactic_sequence("ahhsk") is False
    assert is_valid_phonotactic_sequence("a''sk") is False

    # 2. *C' constraint (post-consonantal glottal stop)
    assert is_valid_phonotactic_sequence("ak'a") is False
    assert is_valid_phonotactic_sequence("at'a") is False
    assert is_valid_phonotactic_sequence("an'a") is False

    # 3. *ChR constraint (stop + voiceless sonorant)
    assert is_valid_phonotactic_sequence("tnh") is False
    assert is_valid_phonotactic_sequence("thlh") is False
    assert is_valid_phonotactic_sequence("khwh") is False


def test_analyze_phonotactics_full_payload():
    """Verify analyze_phonotactics returns a consistent, immutable PhonotacticAnalysis object."""
    text = "ekwhami"
    analysis = analyze_phonotactics(text)

    assert isinstance(analysis, PhonotacticAnalysis)
    assert analysis.text == text
    assert len(analysis.char_syncope_mask) == len(text)
    assert len(analysis.char_intrusion_mask) == len(text)
    assert len(analysis.token_syncope_mask) == len(analysis.tokens)
    assert len(analysis.token_intrusion_mask) == len(analysis.tokens)


def test_prepare_cherokee_text_matrix_and_masks():
    """Verify prepare_cherokee_text constructs ground_truth_mat, syncope, and intrusion masks."""
    from ctc_segmentation import CtcSegmentationParameters  # type: ignore
    from transcription.alignment.phonotactics import prepare_cherokee_text
    import numpy as np

    char_list = ["a", "e", "i", "o", "u", "v", "k", "w", "h", "m", "t", "s", "'"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        space=" ",
    )

    text = "ekwhami atvhska"
    gt_mat, utt_indices = prepare_cherokee_text(config, text, char_list=char_list)

    assert isinstance(gt_mat, np.ndarray)
    assert gt_mat.ndim == 2
    assert gt_mat.shape[1] == 1
    assert len(utt_indices) == 2  # start and end

    # Check that phonotactic masks were attached to config
    assert hasattr(config, "is_syncope_token")
    assert config.is_syncope_token is not None
    assert len(config.is_syncope_token) == gt_mat.shape[0]

    assert hasattr(config, "is_intrusive_site")
    assert config.is_intrusive_site is not None
    assert len(config.is_intrusive_site) == gt_mat.shape[0]


def test_prepare_cherokee_text_with_token_masks():
    """Verify prepare_cherokee_text respects explicit token_masks to isolate English words."""
    from ctc_segmentation import CtcSegmentationParameters  # type: ignore
    from transcription.alignment.phonotactics import prepare_cherokee_text

    char_list = [
        "a",
        "e",
        "i",
        "o",
        "u",
        "v",
        "k",
        "w",
        "h",
        "m",
        "t",
        "s",
        "d",
        "l",
        "n",
        "g",
        "'",
    ]
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        space=" ",
    )

    # Word 1: 'hsowtsa' (English Soldier, len 7) -> zero masks
    # Word 2: 'adalenisgv' (Cherokee Syllabary, len 10) -> Cherokee phonotactics
    text = "hsowtsa adalenisgv"
    english_masks = ([False] * 7, [False] * 7)
    cherokee_masks = (
        [True, False, False, False, True, False, False, False, False, False],
        [False, False, False, False, False, False, False, True, False, False],
    )
    token_masks = [english_masks, cherokee_masks]

    gt_mat, utt_indices = prepare_cherokee_text(
        config, text, char_list=char_list, token_masks=token_masks
    )

    assert config.is_syncope_token is not None
    assert config.is_intrusive_site is not None
    for idx in range(2, 9):
        assert config.is_syncope_token[idx] == 0
        assert config.is_intrusive_site[idx] == 0

    # Index 10 is the start char 'a' of Cherokee 'adalenisgv' -> syncope = 1
    assert config.is_syncope_token[10] == 1


def test_intrusion_site_mask_post_aspiration_and_coda_laryngeals():
    """Verify that get_intrusion_site_mask licenses coda laryngeals and post-consonantal aspiration on vowels."""
    # ukvwali -> u, k, v, w, a, l, i
    # Vowels following plain sonorants and stops have is_intrusive = True (allowing l -> h -> i -> ukvwalhi)
    tokens = tokenize_phonemes("ukvwali")
    mask = get_intrusion_site_mask("ukvwali", return_char_mask=False)

    token_map = {t.symbol: m for t, m in zip(tokens, mask)}
    assert token_map["k"] is True  # Coda / pre-consonantal before k
    assert token_map["v"] is True  # Post-consonantal aspiration k -> h -> v
    assert token_map["w"] is True  # Pre-consonantal before w
    assert token_map["a"] is True  # Post-consonantal aspiration w -> h -> a
    assert token_map["l"] is True  # Pre-consonantal before l
    assert token_map["i"] is True  # Post-consonantal aspiration l -> h -> i (ukvwalhi)


def test_intrusion_site_mask_blocks_preaspirated_sibilants():
    """Verify that get_intrusion_site_mask prohibits intrusive laryngeals before 'hs' and 'hsk'."""
    word = "kvnohsahsti"
    tokens = tokenize_phonemes(word)
    mask = get_intrusion_site_mask(word, return_char_mask=False)

    for tok, is_intrusive in zip(tokens, mask):
        if tok.symbol in ("hs", "hsk", "hst", "hsl"):
            assert (
                is_intrusive is False
            ), f"Intrusion should be blocked on pre-aspirated '{tok.symbol}'"
        if tok.symbol == "a" and tokens[tokens.index(tok) - 1].symbol == "hs":
            assert (
                is_intrusive is False
            ), "Post-aspiration should be blocked on vowel following 'hs'"


def test_lateral_deaffrication_syncope_mask():
    """Verify that get_syncope_mask marks the leading 't' in 'tl' and 'tlh' clusters as eligible for deletion."""
    # Citation 'hatlv' -> 'h', 'a', 't', 'l', 'v'
    # 't' at index 2 should be marked with is_syncope = True, while 'l' at index 3 is False
    mask_char = get_syncope_mask("hatlv", return_char_mask=True)
    assert (
        mask_char[2] is True
    ), "Leading 't' in 'tl' should be marked as syncope candidate"
    assert (
        mask_char[3] is False
    ), "'l' in 'tl' should NOT be marked as syncope candidate"

    # Citation 'tlehsti' (word-initial lateral affricate) -> 't' is char 0, 'l' is char 1
    mask_init = get_syncope_mask("tlehsti", return_char_mask=True)
    assert (
        mask_init[0] is True
    ), "Word-initial 't' in 'tlh' should be marked as syncope candidate"
    assert (
        mask_init[1] is False
    ), "Word-initial 'l' in 'tlh' should NOT be marked as syncope candidate"
