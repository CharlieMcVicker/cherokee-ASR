# -*- coding: utf-8 -*-
"""
Unit tests for reconciliation.py.
"""

from transcription.alignment.models import (
    AlignedChunk,
    AlignmentOutput,
    WordInterval,
)
from transcription.alignment.reconciliation import (
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_word_intervals,
)


def test_reconcile_word_intervals_empty():
    intervals = [WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0)]
    out = reconcile_word_intervals(intervals, syllabary_text="")
    assert len(out) == 1
    assert out[0].word == "osiyo"
    assert out[0].start_sec == 0.0
    assert out[0].end_sec == 1.0


def test_reconcile_word_intervals_with_syllabary(monkeypatch):
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.align_character_syllable",
        lambda s, e: [("osi", "osi"), ("yo", "yo")],
    )
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.reconcile_phonetics",
        lambda syllabary_text, base_transliteration, emitted_text, aligned_pairs: "osiyo_reconciled",
    )

    intervals = [
        WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0, emitted_word="osiyo")
    ]
    out = reconcile_word_intervals(intervals, syllabary_text="ᎣᏏᏲ")
    assert len(out) == 1
    assert out[0].word == "osiyo_reconciled"
    assert out[0].start_sec == 0.0
    assert out[0].end_sec == 1.0


def test_reconcile_word_intervals_fallback_on_exception(monkeypatch):
    def failing_reconcile(*args, **kwargs):
        raise RuntimeError("Reconciliation failed")

    monkeypatch.setattr(
        "transcription.alignment.reconciliation.reconcile_phonetics",
        failing_reconcile,
    )

    intervals = [WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0)]
    out = reconcile_word_intervals(intervals, syllabary_text="ᎣᏏᏲ")
    assert len(out) == 1
    assert out[0].word == "osiyo"


def test_reconcile_alignment_words_and_by_chunk(monkeypatch):
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.align_character_syllable",
        lambda s, e: [("osi", "osi"), ("yo", "yo")],
    )
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.reconcile_phonetics",
        lambda syllabary_text, base_transliteration, emitted_text, aligned_pairs: "osiyo_reconciled",
    )

    c1 = AlignedChunk(
        chunk_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        words=[WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0)],
    )
    c2 = AlignedChunk(
        chunk_id="c2",
        start_sec=1.0,
        end_sec=2.0,
        words=[WordInterval(word="tohiju", start_sec=1.0, end_sec=2.0)],
    )
    alignment = AlignmentOutput(aligned_chunks=[c1, c2])

    flat_words = reconcile_alignment_words(alignment, {"c1": "ᎣᏏᏲ", "c2": ""})
    assert len(flat_words) == 2
    assert flat_words[0].word == "osiyo_reconciled"
    assert flat_words[1].word == "tohiju"

    chunk_map = reconcile_alignment_by_chunk(alignment, {"c1": "ᎣᏏᏲ", "c2": ""})
    assert len(chunk_map["c1"]) == 1
    assert chunk_map["c1"][0].word == "osiyo_reconciled"
    assert len(chunk_map["c2"]) == 1
    assert chunk_map["c2"][0].word == "tohiju"
