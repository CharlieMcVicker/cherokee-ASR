# -*- coding: utf-8 -*-
"""
test_cherokee_enrichment.py

Unit tests for digohwelisgi.cherokee.enrichment module:
- SyllableAlignmentEngine, SyllableAlignment
- Character and syllable alignment
- reconcile_phonetics (syncopation, aspiration, glottal transfer)
- reconcile_alignment_words and reconcile_alignment_by_chunk
"""

import pytest

from digohwelisgi.core.alignment.models import (
    AlignedChunk,
    AlignmentOutput,
    WordInterval,
)
from digohwelisgi.cherokee.enrichment import (
    SyllableAlignment,
    SyllableAlignmentEngine,
    align_character_syllable,
    align_character_syllable_detailed,
    get_base_transliteration,
    is_cherokee_syllable,
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_phonetics,
    reconcile_word_intervals,
)


def test_is_cherokee_syllable():
    assert is_cherokee_syllable("Ꭳ")
    assert is_cherokee_syllable("Ꮣ")
    assert not is_cherokee_syllable("a")
    assert not is_cherokee_syllable(" ")


def test_get_base_transliteration():
    assert get_base_transliteration("ᎠᏓᎴᏂᏍᎬ") == "atalenihskv"


def test_align_character_syllable():
    pairs = align_character_syllable("ᎠᏓᎴᏂᏍᎬ", "ataleniskv")
    assert len(pairs) == len("ᎠᏓᎴᏂᏍᎬ")
    assert "".join(p[1] for p in pairs) == "ataleniskv"


def test_align_character_syllable_detailed():
    details = align_character_syllable_detailed("Ꭳ ᎠᏓᎴᏂᏍᎬ", "o ataleniskv")
    assert len(details) == len("Ꭳ ᎠᏓᎴᏂᏍᎬ")
    assert isinstance(details[0], SyllableAlignment)
    assert details[0].syllabary_char == "Ꭳ"
    assert details[0].emitted_text == "o"


def test_reconcile_phonetics_syncopation():
    syllabary = "ᎠᏓᎴᏂᏍᎬ"
    base_trans = get_base_transliteration(syllabary)
    aligned_pairs = [
        ("Ꭰ", "a"),
        ("Ꮣ", "ta"),
        ("Ꮄ", "le"),
        ("Ꮒ", "ni"),
        ("Ꮝ", "hs"),
        ("Ꭼ", "k"),
    ]
    enriched = reconcile_phonetics(syllabary, base_trans, "atalenihsk", aligned_pairs)
    assert enriched == "atalenihsk"


def test_reconcile_phonetics_laryngeal_toggles():
    syllabary = "Ꮣ Ꭶ"
    base_trans = "ta ka"
    aligned_pairs = [
        ("Ꮣ", "tha"),
        (" ", " "),
        ("Ꭶ", "kha"),
    ]
    enriched = reconcile_phonetics(syllabary, base_trans, "tha kha", aligned_pairs)
    assert enriched == "tha kha"


def test_syllable_alignment_engine():
    engine = SyllableAlignmentEngine()
    reconciled = engine.reconcile("ᎠᏓᎴᏂᏍᎬ", "ataleniskv")
    assert len(reconciled) > 0
    pairs = engine.align("Ꭳ", "o")
    assert pairs == [("Ꭳ", "o")]


def test_reconcile_alignment_words():
    words = [
        WordInterval(
            word="ataleniskv",
            start_sec=0.0,
            end_sec=1.0,
            confidence=0.9,
            emitted_word="ataleniskv",
        )
    ]
    chunk = AlignedChunk(
        chunk_id="chunk_1",
        emitted_text="ataleniskv",
        start_sec=0.0,
        end_sec=1.0,
        words=words,
    )
    alignment = AlignmentOutput(aligned_chunks=[chunk])
    lookup = {"chunk_1": "ᎠᏓᎴᏂᏍᎬ"}

    reconciled_flat = reconcile_alignment_words(alignment, lookup)
    assert len(reconciled_flat) == 1
    assert reconciled_flat[0].word == "atalenihskv"

    by_chunk = reconcile_alignment_by_chunk(alignment, lookup)
    assert "chunk_1" in by_chunk
    assert len(by_chunk["chunk_1"]) == 1
