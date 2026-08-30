# -*- coding: utf-8 -*-
"""
aligner.py

Backward-compatibility facade for ASR CTC emissions extraction & DTW Alignment Engine.
Delegates to the new ports & adapters transcription.alignment core engine.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple, Sequence, Union
import numpy as np

from transcription.alignment.domain.models import (
    TokenEmission,
    TextChunk,
    AlignmentOutput,
    AlignedChunk,
)
from transcription.alignment.core.sliding_window import SlidingWindowDTWAligner
from transcription.alignment.core.word_aligner import NeedlemanWunschWordAligner
from transcription.alignment.adapters.inbound import (
    BibleMetadataVerseAdapter,
    GenericChunkListAdapter,
)
from transcription.alignment.strategies.distance_metrics import (
    DefaultCERDistanceMetric,
)
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
)
from transcription.alignment.strategies.reconciliation import (
    CherokeeSyllabaryReconciliationStrategy,
)
from transcription.timestamping.prepare_ground_truth import normalize_text_for_alignment
from transcription.timestamping.audio_segmenter import segment_long_audio, AudioChunk
from transcription.alignment.ports.protocols import ASREmissionsExtractor
from transcription.models.asr_model import CherokeeASRModel


@dataclass
class WordInterval:
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    flagged: bool = False
    cherokee_syllabary: str = ""
    reconciled_word: str = ""
    emitted_word: str = ""


@dataclass
class VerseInterval:
    line_id: str
    cherokee_syllabary: str
    raw_phonetic: str
    english: str
    start_sec: float
    end_sec: float
    words: List[WordInterval] = field(default_factory=list)
    cer: float = 1.0
    emitted_text: str = ""


@dataclass
class AlignmentMetrics:
    total_verses: int
    matched_verses: int
    matched_verse_ratio: float
    overall_cer: float
    mean_verse_cer: float
    total_ground_truth_chars: int
    total_emitted_chars: int


@dataclass
class AlignmentResult:
    audio_source: str
    verses: List[VerseInterval]
    metrics: Optional[AlignmentMetrics] = None
    raw_tokens: List[Dict[str, Any]] = field(default_factory=list)


def compute_trigram_edit_cost(emissions_str: str, ground_truth_str: str) -> float:
    """Computes CER edit cost between emitted tokens and normalized ground truth using jiwer."""
    metric = DefaultCERDistanceMetric()
    return metric.compute_cost(hypothesis=emissions_str, reference=ground_truth_str)


def compute_alignment_metrics(alignment_result: AlignmentResult) -> AlignmentMetrics:
    """
    Computes segment alignment quality metrics exclusively across matched verses.
    """
    verses = alignment_result.verses
    total_verses = len(verses)
    if total_verses == 0:
        metrics = AlignmentMetrics(
            total_verses=0,
            matched_verses=0,
            matched_verse_ratio=0.0,
            overall_cer=1.0,
            mean_verse_cer=1.0,
            total_ground_truth_chars=0,
            total_emitted_chars=0,
        )
        alignment_result.metrics = metrics
        return metrics

    matched_verse_list = [v for v in verses if v.words and (v.end_sec > v.start_sec)]
    matched_verses_count = len(matched_verse_list)
    matched_ratio = round(matched_verses_count / total_verses, 4)

    verse_cers = [v.cer for v in matched_verse_list]
    mean_verse_cer = round(float(np.mean(verse_cers)), 4) if verse_cers else 1.0

    # CER exclusively across matched verses
    matched_gt = []
    matched_emitted = []
    preprocessor = CherokeePhoneticPreprocessor()
    for v in matched_verse_list:
        gt_norm = preprocessor.normalize(v.raw_phonetic)
        em_norm = preprocessor.normalize(v.emitted_text)
        if gt_norm:
            matched_gt.append(gt_norm)
            matched_emitted.append(em_norm)

    concat_gt = " ".join(matched_gt)
    concat_emitted = " ".join(matched_emitted)

    overall_cer = (
        compute_trigram_edit_cost(concat_emitted, concat_gt) if concat_gt else 1.0
    )

    metrics = AlignmentMetrics(
        total_verses=total_verses,
        matched_verses=matched_verses_count,
        matched_verse_ratio=matched_ratio,
        overall_cer=round(overall_cer, 4),
        mean_verse_cer=mean_verse_cer,
        total_ground_truth_chars=len(concat_gt),
        total_emitted_chars=len(concat_emitted),
    )
    alignment_result.metrics = metrics
    return metrics


def _align_words_char_range(
    raw_words: List[str],
    matched_tokens: List[Dict[str, Any]],
    raw_syllabary_words: Optional[List[str]] = None,
) -> List[WordInterval]:
    """
    Aligns ground-truth words to matched emission tokens using NeedlemanWunschWordAligner.
    """
    if not raw_words or not matched_tokens:
        return []

    token_emissions = [
        TokenEmission(
            word=t.get("word", ""),
            start_sec=t.get("start_time", t.get("start_sec", 0.0)),
            end_sec=t.get("end_time", t.get("end_sec", 0.0)),
            confidence=t.get("confidence", 1.0),
        )
        for t in matched_tokens
    ]

    word_aligner = NeedlemanWunschWordAligner()
    domain_words = word_aligner.align_words(
        raw_words=raw_words,
        matched_tokens=token_emissions,
        raw_syllabary_words=raw_syllabary_words,
    )

    return [
        WordInterval(
            word=w.word,
            start_sec=w.start_sec,
            end_sec=w.end_sec,
            confidence=w.confidence,
            flagged=w.flagged,
            cherokee_syllabary=w.syllabary or "",
            reconciled_word=w.reconciled_word or "",
            emitted_word=w.emitted_word or "",
        )
        for w in domain_words
    ]


def align_tokens_to_verses(
    token_emissions: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    audio_source: str = "",
    reconcile: bool = False,
) -> AlignmentResult:
    """
    Aligns raw CTC token emissions with global timestamps to ground-truth verses
    using SlidingWindowDTWAligner and ports & adapters domain entities.

    Args:
        token_emissions: List of word dicts with start_time, end_time, and word text.
        verses: List of parsed ground-truth verse dicts from prepare_ground_truth.
        audio_source: Optional identifier or path string for metadata.
        reconcile: If True, performs phonological syllabary/ASR reconciliation on aligned words.

    Returns:
        AlignmentResult data structure with populated verse & word interval timestamps and CER metrics.
    """
    # 1. Convert input token emissions to domain TokenEmission dataclass list
    domain_emissions: List[TokenEmission] = []
    for t in token_emissions or []:
        domain_emissions.append(
            TokenEmission(
                word=t.get("word", ""),
                start_sec=t.get("start_time", t.get("start_sec", 0.0)),
                end_sec=t.get("end_time", t.get("end_sec", 0.0)),
                confidence=t.get("confidence", 1.0),
            )
        )

    # 2. Convert ground truth verses to domain TextChunk dataclass list
    inbound_adapter = BibleMetadataVerseAdapter()
    chunks = inbound_adapter.load_chunks_from_dict(verses)

    # 3. Configure domain engine & strategies
    engine = SlidingWindowDTWAligner()
    reconciliation_strategy = (
        CherokeeSyllabaryReconciliationStrategy() if reconcile else None
    )

    # 4. Execute core alignment
    domain_output: AlignmentOutput = engine.align_chunks(
        emissions=domain_emissions,
        chunks=chunks,
        reconciliation_strategy=reconciliation_strategy,
        audio_source=audio_source,
    )

    # 5. Map AlignedChunks back to legacy VerseInterval and AlignmentResult
    legacy_verses = inbound_adapter.to_verse_intervals(domain_output)
    result = AlignmentResult(
        audio_source=audio_source,
        verses=legacy_verses,
        raw_tokens=token_emissions,
    )

    # 6. Compute metrics
    compute_alignment_metrics(result)
    return result


def align_emissions_to_text(
    token_emissions: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    audio_source: str = "",
    reconcile: bool = False,
) -> AlignmentResult:
    """
    Lightweight CPU-based alignment function taking pre-computed emitted tokens
    and ground-truth verse dictionary objects in memory.

    Args:
        token_emissions: List of word dicts with start_time, end_time, and word text.
        verses: List of ground-truth verse/segment dicts.
        audio_source: Optional identifier or path string for metadata.
        reconcile: If True, performs phonological syllabary/ASR reconciliation on aligned words.

    Returns:
        AlignmentResult data structure.
    """
    return align_tokens_to_verses(
        token_emissions, verses, audio_source=audio_source, reconcile=reconcile
    )


def align_audio_segment(
    audio_input: Any,
    verses: List[Dict[str, Any]],
    extractor: ASREmissionsExtractor,
    audio_source: str = "",
    reconcile: bool = False,
) -> AlignmentResult:
    """
    Exposes audio/emissions alignment for in-memory execution using typed extractors.
    Extracts token emissions via the provided extractor adapter and passes them
    to the pure in-memory align_emissions_to_text function.

    Args:
        audio_input: Audio file path, AudioSegment, or raw audio waveform sample array.
        verses: List of ground-truth verse/segment dicts.
        extractor: ASREmissionsExtractor strategy.
        audio_source: Optional label or path string for output metadata.
        reconcile: If True, performs phonological syllabary/ASR reconciliation on aligned words.

    Returns:
        AlignmentResult data structure.
    """
    emissions = extractor.extract(audio_input)
    raw_tokens = [
        {
            "word": e.word,
            "start_time": e.start_sec,
            "end_time": e.end_sec,
            "confidence": e.confidence,
        }
        for e in emissions
    ]

    return align_emissions_to_text(
        token_emissions=raw_tokens,
        verses=verses,
        audio_source=audio_source,
        reconcile=reconcile,
    )


def create_audio_aligner(
    extractor: ASREmissionsExtractor,
    reconcile: bool = False,
) -> Callable[..., AlignmentResult]:
    """
    Curried factory returning a reusable audio alignment callable configured with
    the provided extractor and alignment settings.
    """

    def aligner(
        audio_input: Any,
        verses: List[Dict[str, Any]],
        audio_source: str = "",
    ) -> AlignmentResult:
        return align_audio_segment(
            audio_input=audio_input,
            verses=verses,
            extractor=extractor,
            audio_source=audio_source,
            reconcile=reconcile,
        )

    return aligner
