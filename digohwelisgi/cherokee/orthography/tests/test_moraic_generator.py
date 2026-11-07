# -*- coding: utf-8 -*-
"""
test_moraic_generator.py

Unit tests for Phase 2 Moraic Tone Target Generator & Dual-Stream Sequence Reconciliation.
"""

import csv
from pathlib import Path
import pytest

from digohwelisgi.cherokee.orthography import (
    MORAIC_TONE_VOCAB,
    moraic_tokens_to_pitch_contour,
    reconcile_dual_stream,
    to_canonical_toned_tth,
    to_moraic_tone_tokens,
)


def test_short_vowel_mora_mapping() -> None:
    """Assert short low -> 2_1, short high -> 3_1."""
    cv_low, tone_low = to_moraic_tone_tokens("a")
    assert cv_low == "a"
    assert tone_low == "2_1"

    cv_high, tone_high = to_moraic_tone_tokens("á")
    assert cv_high == "a"
    assert tone_high == "3_1"


def test_long_vowel_and_contour_mora_mapping() -> None:
    """
    Verify long vowels and contour mora mappings:
    - Level low 22 -> 2_1 2_2
    - Level high 33 -> 3_1 3_2
    - Low-fall 21 -> 2_1 1_2
    - Rising 23 -> 2_1 3_2
    - Falling 32 -> 3_1 2_2
    - Superhigh 34 -> 3_1 4_2
    """
    cases = [
        ("a:", "2_1 2_2"),
        ("a22", "2_1 2_2"),
        ("á:", "3_1 3_2"),
        ("a33", "3_1 3_2"),
        ("à:", "2_1 1_2"),
        ("a21", "2_1 1_2"),
        ("ǎ:", "2_1 3_2"),
        ("a23", "2_1 3_2"),
        ("â:", "3_1 2_2"),
        ("a32", "3_1 2_2"),
        ("a̋:", "3_1 4_2"),
        ("a34", "3_1 4_2"),
    ]
    for raw, expected_mora in cases:
        cv, tone = to_moraic_tone_tokens(raw)
        assert cv == "a"
        assert tone == expected_mora


def test_onset_separator_rules() -> None:
    """
    Verify onset / consonant separator C rules:
    - Words with consonant cluster: skwa -> C followed by vowel moras.
    - Words starting with vowel: ama -> no leading C (2_1 C 2_1).
    - Glottal stop onsets: 'a -> C followed by vowel moras.
    """
    cv_cluster, tone_cluster = to_moraic_tone_tokens("skwa")
    assert cv_cluster == "skwa"
    assert tone_cluster == "C 2_1"

    cv_vowel, tone_vowel = to_moraic_tone_tokens("ama")
    assert cv_vowel == "ama"
    assert tone_vowel == "2_1 C 2_1"

    cv_glottal, tone_glottal = to_moraic_tone_tokens("'a")
    assert cv_glottal == "'a"
    assert tone_glottal == "C 2_1"


def test_reconciliation_roundtrip() -> None:
    """
    Feed curated Feeling dictionary sentences through to_canonical_toned_tth ->
    to_moraic_tone_tokens -> reconcile_dual_stream (both dp and fast modes)
    and assert 100% exact round-trip match across all valid sentences in sentence_audio.csv.
    """
    csv_path = Path("data/training/processed/sentence_audio.csv")
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [
            {
                "phonetic": "ná *ani:chű:ja* sgwà:hle̋:sdi dà:ná:ne:lo:hv́sga",
                "syllabary": "Ꮎ *ᎠᏂᏧᏣ* ᏍᏆᏞᏍᏗ ᏓᎾᏁᎶᎲᏍᎦ.",
            },
            {
                "phonetic": "jitǎ:ga *ani:chv:ya* u:gitsde̋:yi à:ná:yhgó:’i",
                "syllabary": "ᏥᏔᎦ *ᎠᏂᏨᏯ* ᎤᎩᏥᏍᏕᏱ ᎠᎾᏴᎪᎢ.",
            },
        ]

    checked = 0
    for row in rows:
        canon, dropped = to_canonical_toned_tth(
            row["phonetic"], syllabary=row.get("syllabary")
        )
        if dropped or not canon or "f" in canon or "b" in canon:
            continue
        cv_text, tone_tokens = to_moraic_tone_tokens(canon)
        for tok in tone_tokens.split():
            assert tok in MORAIC_TONE_VOCAB
        assert reconcile_dual_stream(cv_text, tone_tokens, mode="dp") == canon
        assert reconcile_dual_stream(cv_text, tone_tokens, mode="fast") == canon
        checked += 1

    assert checked >= 50


def test_moraic_tokens_to_pitch_contour() -> None:
    contour = moraic_tokens_to_pitch_contour("C 2_1 | C 3_1 4_2", step_sec=0.1)
    assert contour == [(0.1, 2.0), (0.4, 3.0), (0.5, 4.0)]
