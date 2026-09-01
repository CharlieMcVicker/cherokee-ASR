# -*- coding: utf-8 -*-
"""
reconciliation.py

Pure reconciliation function mapping ground-truth Cherokee syllabary against aligned words
and enriching phonetic representations using acoustic ASR emissions.
"""

from typing import Dict
from transcription.alignment.models import AlignmentOutput
from transcription.syllabary_enrichment import (
    align_character_syllable,
    reconcile_phonetics,
)


def reconcile_alignment(
    alignment: AlignmentOutput, syllabary_lookup: Dict[str, str]
) -> AlignmentOutput:
    """
    Reconciles canonical Cherokee syllabary with emitted ASR tokens across all aligned chunks.
    Populates w.reconciled_word for each WordInterval where syllabary text is available.

    Args:
        alignment: AlignmentOutput object containing aligned chunks.
        syllabary_lookup: Mapping from chunk_id to Cherokee syllabary text string.

    Returns:
        The updated AlignmentOutput with populated reconciled_word fields.
    """
    for chunk in alignment.aligned_chunks:
        syll_text = syllabary_lookup.get(chunk.chunk_id, "").strip()
        if not syll_text or not chunk.words:
            for w in chunk.words:
                if w.reconciled_word is None:
                    w.reconciled_word = w.word
            continue

        syll_words = [sw for sw in syll_text.split() if sw]

        if len(syll_words) == len(chunk.words):
            for w_idx, w in enumerate(chunk.words):
                target_syll = syll_words[w_idx]
                target_emitted = w.emitted_word or w.word
                try:
                    align_pairs = align_character_syllable(target_syll, target_emitted)
                    base_trans = "".join([pair[0] for pair in align_pairs])
                    rec_word = reconcile_phonetics(
                        syllabary_text=target_syll,
                        base_transliteration=base_trans,
                        emitted_text=target_emitted,
                        aligned_pairs=align_pairs,
                    )
                    w.reconciled_word = rec_word
                except Exception:
                    w.reconciled_word = w.word
        else:
            for w_idx, w in enumerate(chunk.words):
                target_syll = (
                    syll_words[w_idx] if w_idx < len(syll_words) else syll_text
                )
                target_emitted = w.emitted_word or w.word
                try:
                    align_pairs = align_character_syllable(target_syll, target_emitted)
                    base_trans = "".join([pair[0] for pair in align_pairs])
                    rec_word = reconcile_phonetics(
                        syllabary_text=target_syll,
                        base_transliteration=base_trans,
                        emitted_text=target_emitted,
                        aligned_pairs=align_pairs,
                    )
                    w.reconciled_word = rec_word
                except Exception:
                    w.reconciled_word = w.word

    return alignment
