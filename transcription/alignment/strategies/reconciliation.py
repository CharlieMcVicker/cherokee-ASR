"""
Reconciliation strategy implementation integrating syllabary enrichment phonetics.
"""

from typing import List, Tuple
from transcription.alignment.ports.protocols import ReconciliationStrategy


class CherokeeSyllabaryReconciliationStrategy:
    """
    Reconciles canonical Cherokee syllabary with emitted ASR tokens using
    syllabary enrichment character-syllable alignment and phonological reconciliation.
    """

    def reconcile(
        self, syllabary_text: str, emitted_text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        if not syllabary_text or not emitted_text:
            return emitted_text, []

        from transcription.syllabary_enrichment import (
            align_character_syllable,
            reconcile_phonetics,
        )

        align_pairs = align_character_syllable(syllabary_text, emitted_text)
        base_trans = "".join([pair[0] for pair in align_pairs])
        try:
            reconciled = reconcile_phonetics(
                syllabary_text=syllabary_text,
                base_transliteration=base_trans,
                emitted_text=emitted_text,
                aligned_pairs=align_pairs,
            )
            return reconciled, align_pairs
        except Exception:
            return emitted_text, align_pairs
