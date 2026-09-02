# -*- coding: utf-8 -*-
"""
reconciliation.py

Pure reconciliation functions mapping ground-truth Cherokee syllabary against aligned words
and enriching phonetic representations using acoustic ASR emissions into new WordInterval collections.
"""

from typing import Dict, List, Sequence
from transcription.alignment.models import AlignmentOutput, WordInterval
from transcription.syllabary_enrichment import (
    align_character_syllable,
    reconcile_phonetics,
)


def reconcile_word_intervals(
    words: Sequence[WordInterval],
    syllabary_text: str,
) -> List[WordInterval]:
    """
    Pure mapping: takes word intervals and syllabary reference text,
    returning a new list of WordIntervals where `word` contains the reconciled phonetics.

    Args:
        words: Sequence of WordInterval objects to reconcile.
        syllabary_text: Ground-truth Cherokee syllabary text string.

    Returns:
        New List of WordInterval instances with reconciled word strings.
    """
    syll_text = syllabary_text.strip()
    if not syll_text or not words:
        return [
            WordInterval(
                word=w.word,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                confidence=w.confidence,
                flagged=w.flagged,
                emitted_word=w.emitted_word,
            )
            for w in words
        ]

    syll_words = [sw for sw in syll_text.split() if sw]
    reconciled: List[WordInterval] = []

    for w_idx, w in enumerate(words):
        if len(syll_words) == len(words):
            target_syll = syll_words[w_idx]
        else:
            target_syll = syll_words[w_idx] if w_idx < len(syll_words) else syll_text

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
        except Exception:
            rec_word = w.word

        reconciled.append(
            WordInterval(
                word=rec_word,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                confidence=w.confidence,
                flagged=w.flagged,
                emitted_word=w.emitted_word,
            )
        )

    return reconciled


def reconcile_alignment_words(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> List[WordInterval]:
    """
    Reconciles all words across chunks in an AlignmentOutput,
    returning a single flattened list of reconciled WordIntervals.

    Args:
        alignment: AlignmentOutput object containing aligned chunks.
        syllabary_lookup: Mapping from chunk_id to Cherokee syllabary text string.

    Returns:
        Flattened List of reconciled WordInterval objects.
    """
    all_reconciled: List[WordInterval] = []
    for chunk in alignment.aligned_chunks:
        syll_text = syllabary_lookup.get(chunk.chunk_id, "")
        all_reconciled.extend(reconcile_word_intervals(chunk.words, syll_text))
    return all_reconciled


def reconcile_alignment_by_chunk(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> Dict[str, List[WordInterval]]:
    """
    Reconciles words grouped by chunk_id.

    Args:
        alignment: AlignmentOutput object containing aligned chunks.
        syllabary_lookup: Mapping from chunk_id to Cherokee syllabary text string.

    Returns:
        Mapping from chunk_id to List of reconciled WordInterval objects.
    """
    result: Dict[str, List[WordInterval]] = {}
    for chunk in alignment.aligned_chunks:
        syll_text = syllabary_lookup.get(chunk.chunk_id, "")
        result[chunk.chunk_id] = reconcile_word_intervals(chunk.words, syll_text)
    return result
