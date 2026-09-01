# -*- coding: utf-8 -*-
"""
Unit tests for reconciliation.py.
"""

from transcription.alignment.models import (
    AlignedChunk,
    AlignmentOutput,
    WordInterval,
)
from transcription.alignment.reconciliation import reconcile_alignment


def test_reconcile_alignment_empty_lookup():
    chunk = AlignedChunk(
        chunk_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        words=[WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0)],
    )
    alignment = AlignmentOutput(aligned_chunks=[chunk])
    out = reconcile_alignment(alignment, syllabary_lookup={})
    assert out.aligned_chunks[0].words[0].reconciled_word == "osiyo"


def test_reconcile_alignment_with_syllabary(monkeypatch):
    # Mock syllabary reconciliation functions
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.align_character_syllable",
        lambda s, e: [("osi", "osi"), ("yo", "yo")],
    )
    monkeypatch.setattr(
        "transcription.alignment.reconciliation.reconcile_phonetics",
        lambda syllabary_text, base_transliteration, emitted_text, aligned_pairs: "osiyo_reconciled",
    )

    chunk = AlignedChunk(
        chunk_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        words=[
            WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0, emitted_word="osiyo")
        ],
    )
    alignment = AlignmentOutput(aligned_chunks=[chunk])
    out = reconcile_alignment(alignment, syllabary_lookup={"c1": "ᎣᏏᏲ"})
    assert out.aligned_chunks[0].words[0].reconciled_word == "osiyo_reconciled"


def test_reconcile_alignment_with_exception_fallback(monkeypatch):
    def failing_reconcile(*args, **kwargs):
        raise RuntimeError("Reconciliation failed")

    monkeypatch.setattr(
        "transcription.alignment.reconciliation.reconcile_phonetics",
        failing_reconcile,
    )

    chunk = AlignedChunk(
        chunk_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        words=[WordInterval(word="osiyo", start_sec=0.0, end_sec=1.0)],
    )
    alignment = AlignmentOutput(aligned_chunks=[chunk])
    out = reconcile_alignment(alignment, syllabary_lookup={"c1": "ᎣᏏᏲ"})
    assert out.aligned_chunks[0].words[0].reconciled_word == "osiyo"
