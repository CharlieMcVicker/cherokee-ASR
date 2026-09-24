# -*- coding: utf-8 -*-
"""
transcription.pipelines.enrichment.pipeline module.

End-to-end Syllabary Phonetic Enrichment Pipeline.
Orchestrates:
1. Ingestion of Cherokee Syllabary text / manifests.
2. Acoustic inference via CherokeeASRModel / ModelOutput universal currency to obtain emitted phonetics.
3. Fine-grained character and syllable level alignment between ground truth Cherokee Syllabary
   and ASR emitted phonetics.
4. Phonological enrichment rule merger applying syncopation, glottal/pre-aspiration transfer,
   and laryngeal toggles while preserving Cherokee Syllabary as the immutable structural anchor.
5. Evaluation metrics calculation (raw vs. reconciled CER, relative improvement) and manifest exports.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydub import AudioSegment

from transcription.cherokee.enrichment import (
    SyllableAlignment,
    align_character_syllable,
    align_character_syllable_detailed,
    get_base_transliteration,
    is_cherokee_syllable,
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_phonetics,
    reconcile_word_intervals,
)
from transcription.cherokee.models import CherokeeASRModel
from transcription.core.alignment.models import AlignmentOutput, WordInterval
from transcription.core.models.output import ModelOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentRecord:
    """
    Data structure representing a single record processed through the enrichment pipeline.
    """

    record_id: str
    syllabary_text: str
    base_transliteration: str
    emitted_text: str
    reconciled_text: str
    aligned_pairs: List[Tuple[str, str]]
    audio_path: Optional[str] = None
    target_phonetics: Optional[str] = None
    split: Optional[str] = None
    raw_cer: Optional[float] = None
    reconciled_cer: Optional[float] = None
    delta_cer: Optional[float] = None


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) via normalized Levenshtein distance.
    """
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    len_r, len_h = len(ref), len(hyp)
    dp = [[0] * (len_h + 1) for _ in range(len_r + 1)]
    for i in range(len_r + 1):
        dp[i][0] = i
    for j in range(len_h + 1):
        dp[0][j] = j

    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return float(dp[len_r][len_h]) / float(len_r)


def calculate_relative_improvement(raw_cer: float, reconciled_cer: float) -> float:
    """
    Calculate percentage relative CER improvement: (raw - reconciled) / raw * 100.
    """
    if raw_cer <= 0.0:
        return 0.0
    return ((raw_cer - reconciled_cer) / raw_cer) * 100.0


class EnrichmentPipeline:
    """
    Orchestrator for Cherokee Syllabary Phonetic Reconciliation.

    Composes acoustic ASR observations with immutable Syllabary ground truth
    to reconcile surface phonetic realizations (vowel deletion/syncopation,
    aspiration/glottal transfer, laryngeal toggles).
    """

    def __init__(
        self,
        asr_model: Optional[CherokeeASRModel] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.asr_model = asr_model
        self.model_path = model_path

    def _get_model(self) -> CherokeeASRModel:
        if self.asr_model is None:
            token = os.environ.get("HF_TOKEN", None)
            self.asr_model = CherokeeASRModel.from_pretrained_or_best(
                path_or_repo=self.model_path,
                token=token,
            )
        return self.asr_model

    def enrich_text(
        self,
        syllabary_text: str,
        emitted_text: str,
        base_transliteration: Optional[str] = None,
    ) -> EnrichmentRecord:
        """
        Reconciles a single Cherokee Syllabary string against emitted ASR phonetics.

        Args:
            syllabary_text: Native Cherokee Unicode syllabary glyphs.
            emitted_text: Acoustic ASR emission hypothesis.
            base_transliteration: Optional precomputed base transliteration.

        Returns:
            EnrichmentRecord containing reconciled text and character-level alignments.
        """
        base_trans = (
            base_transliteration
            if base_transliteration is not None
            else get_base_transliteration(syllabary_text)
        )
        aligned_pairs = align_character_syllable(syllabary_text, emitted_text)
        reconciled = reconcile_phonetics(
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            aligned_pairs=aligned_pairs,
        )
        return EnrichmentRecord(
            record_id="",
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            reconciled_text=reconciled,
            aligned_pairs=aligned_pairs,
        )

    def enrich_audio(
        self,
        audio: Union[str, Path, AudioSegment, np.ndarray, ModelOutput],
        syllabary_text: str,
        base_transliteration: Optional[str] = None,
        record_id: str = "",
    ) -> EnrichmentRecord:
        """
        Transcribes audio via ASR model and reconciles against syllabary text.

        Args:
            audio: Audio path, AudioSegment, waveform array, or precomputed ModelOutput.
            syllabary_text: Native Cherokee Syllabary text.
            base_transliteration: Optional precomputed base transliteration.
            record_id: Optional identifier.

        Returns:
            EnrichmentRecord with emitted and reconciled text.
        """
        model_out: ModelOutput
        if isinstance(audio, ModelOutput):
            model_out = audio
        else:
            model = self._get_model()
            if isinstance(audio, AudioSegment):
                audio_arr = (
                    np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                )
                model_out = model.infer(audio_arr)
            else:
                model_out = model.infer(audio)

        emitted_text = model_out.decode_greedy()
        base_trans = (
            base_transliteration
            if base_transliteration is not None
            else get_base_transliteration(syllabary_text)
        )
        aligned_pairs = align_character_syllable(syllabary_text, emitted_text)
        reconciled = reconcile_phonetics(
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            aligned_pairs=aligned_pairs,
        )
        return EnrichmentRecord(
            record_id=record_id,
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            reconciled_text=reconciled,
            aligned_pairs=aligned_pairs,
            audio_path=str(audio) if isinstance(audio, (str, Path)) else None,
        )

    def reconcile_alignment(
        self,
        alignment: AlignmentOutput,
        syllabary_lookup: Dict[str, str],
    ) -> List[WordInterval]:
        """
        Reconciles word intervals in an AlignmentOutput against syllabary lookup.
        """
        return reconcile_alignment_words(alignment, syllabary_lookup)

    def run_manifest(
        self,
        records: Sequence[Dict[str, Any]],
        output_cache_path: Optional[Union[str, Path]] = None,
        evaluate: bool = True,
    ) -> Tuple[List[EnrichmentRecord], Dict[str, Any]]:
        """
        Processes a collection of manifest records, extracting ASR emissions
        (or using precomputed emissions), performing reconciliation, and evaluating CER.

        Args:
            records: List of dictionaries with keys:
                     - syllabary / syllabary_text / text
                     - emitted_text (optional)
                     - audio_filepath / audio_path / path (optional if emitted_text given)
                     - target / target_phonetics (optional, for evaluation)
                     - split (optional)
            output_cache_path: Optional JSON path to export results.
            evaluate: Whether to compute evaluation metrics.

        Returns:
            Tuple of (List of EnrichmentRecords, Summary Dict).
        """
        results: List[EnrichmentRecord] = []
        needs_asr: List[Tuple[int, Optional[str]]] = []

        for idx, rec in enumerate(records):
            syll = (
                rec.get("syllabary_text")
                or rec.get("cherokee_text")
                or rec.get("syllabary")
                or rec.get("text")
                or ""
            )
            emitted_val: Optional[str] = rec.get("emitted_text")
            raw_audio_path = (
                rec.get("audio_filepath")
                or rec.get("audio_path")
                or rec.get("file_path")
                or rec.get("path")
            )
            audio_path = str(raw_audio_path) if raw_audio_path is not None else None
            if emitted_val is None:
                needs_asr.append((idx, audio_path))

        # Perform batched/sequential ASR if needed
        asr_emissions: Dict[int, str] = {}
        if needs_asr:
            model = self._get_model()
            for idx, a_path in needs_asr:
                if not a_path or not os.path.exists(str(a_path)):
                    asr_emissions[idx] = ""
                else:
                    out = model.infer(a_path)
                    emitted_str = str(out.decode_greedy())
                    asr_emissions[idx] = emitted_str

        for idx, rec in enumerate(records):
            rec_id = (
                rec.get("id")
                or rec.get("sentence_id")
                or rec.get("record_id")
                or str(idx)
            )
            syll = (
                rec.get("syllabary_text")
                or rec.get("cherokee_text")
                or rec.get("syllabary")
                or rec.get("text")
                or ""
            )
            audio_path = (
                rec.get("audio_filepath")
                or rec.get("audio_path")
                or rec.get("file_path")
                or rec.get("path")
            )
            raw_emitted = (
                rec.get("emitted_text")
                if rec.get("emitted_text") is not None
                else asr_emissions.get(idx, "")
            )
            emitted: str = str(raw_emitted) if raw_emitted is not None else ""
            raw_base_trans = rec.get("base_transliteration")
            base_trans: str = (
                str(raw_base_trans)
                if raw_base_trans is not None
                else get_base_transliteration(syll)
            )
            raw_target = rec.get("target") or rec.get("target_phonetics")
            target: Optional[str] = str(raw_target) if raw_target is not None else None
            split = rec.get("split")

            aligned_pairs = align_character_syllable(syll, emitted)
            reconciled = reconcile_phonetics(
                syllabary_text=syll,
                base_transliteration=base_trans,
                emitted_text=emitted,
                aligned_pairs=aligned_pairs,
            )

            raw_cer: Optional[float] = None
            rec_cer: Optional[float] = None
            delta_cer: Optional[float] = None
            if evaluate and target is not None:
                raw_cer = calculate_cer(target, emitted)
                rec_cer = calculate_cer(target, reconciled)
                delta_cer = calculate_relative_improvement(raw_cer, rec_cer)

            item = EnrichmentRecord(
                record_id=rec_id,
                syllabary_text=syll,
                base_transliteration=base_trans,
                emitted_text=emitted,
                reconciled_text=reconciled,
                aligned_pairs=aligned_pairs,
                audio_path=audio_path,
                target_phonetics=target,
                split=split,
                raw_cer=raw_cer,
                reconciled_cer=rec_cer,
                delta_cer=delta_cer,
            )
            results.append(item)

        # Build summary
        summary: Dict[str, Any] = {}
        if evaluate:
            splits_data: Dict[str, List[EnrichmentRecord]] = {}
            for r in results:
                s_name = r.split or "unspecified"
                splits_data.setdefault(s_name, []).append(r)

            for s_name, s_records in splits_data.items():
                evaluated = [
                    r
                    for r in s_records
                    if r.raw_cer is not None and r.reconciled_cer is not None
                ]
                if evaluated:
                    raw_vals: List[float] = [
                        r.raw_cer for r in evaluated if r.raw_cer is not None
                    ]
                    rec_vals: List[float] = [
                        r.reconciled_cer
                        for r in evaluated
                        if r.reconciled_cer is not None
                    ]
                    mean_raw = float(np.mean(raw_vals))
                    mean_rec = float(np.mean(rec_vals))
                    summary[s_name] = {
                        "count": len(evaluated),
                        "raw_cer": round(mean_raw, 4),
                        "reconciled_cer": round(mean_rec, 4),
                        "delta_cer": round(
                            calculate_relative_improvement(mean_raw, mean_rec), 2
                        ),
                    }

            all_eval = [
                r
                for r in results
                if r.raw_cer is not None and r.reconciled_cer is not None
            ]
            if all_eval:
                all_raw_vals: List[float] = [
                    r.raw_cer for r in all_eval if r.raw_cer is not None
                ]
                all_rec_vals: List[float] = [
                    r.reconciled_cer for r in all_eval if r.reconciled_cer is not None
                ]
                overall_raw = float(np.mean(all_raw_vals))
                overall_rec = float(np.mean(all_rec_vals))
                summary["overall"] = {
                    "count": len(all_eval),
                    "raw_cer": round(overall_raw, 4),
                    "reconciled_cer": round(overall_rec, 4),
                    "delta_cer": round(
                        calculate_relative_improvement(overall_raw, overall_rec), 2
                    ),
                }

        if output_cache_path is not None:
            out_p = Path(output_cache_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            export_payload = {
                "summary": summary,
                "records": [asdict(r) for r in results],
            }
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(export_payload, f, ensure_ascii=False, indent=2)

        return results, summary


def enrich_syllabary(
    syllabary_text: str,
    emitted_text: str,
    base_transliteration: Optional[str] = None,
) -> str:
    """
    Pure procedural convenience function to enrich Cherokee Syllabary text
    against an ASR hypothesis.
    """
    pipeline = EnrichmentPipeline()
    return pipeline.enrich_text(
        syllabary_text=syllabary_text,
        emitted_text=emitted_text,
        base_transliteration=base_transliteration,
    ).reconciled_text


__all__ = [
    "EnrichmentPipeline",
    "EnrichmentRecord",
    "calculate_cer",
    "calculate_relative_improvement",
    "enrich_syllabary",
]
