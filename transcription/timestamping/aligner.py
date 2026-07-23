# -*- coding: utf-8 -*-
"""
aligner.py

ASR CTC emissions extraction & Trigram Sliding-Window DTW Alignment Engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from jiwer import cer as jiwer_cer
import numpy as np

from transcription.timestamping.prepare_ground_truth import normalize_text_for_alignment


@dataclass
class WordInterval:
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    flagged: bool = False


@dataclass
class VerseInterval:
    line_id: str
    cherokee_syllabary: str
    raw_phonetic: str
    english: str
    start_sec: float
    end_sec: float
    words: List[WordInterval] = field(default_factory=list)


@dataclass
class AlignmentResult:
    audio_source: str
    verses: List[VerseInterval]


def compute_trigram_edit_cost(emissions_str: str, ground_truth_str: str) -> float:
    """Computes CER edit cost between emitted tokens and normalized ground truth using jiwer."""
    if not emissions_str or not ground_truth_str:
        return 1.0
    try:
        return float(jiwer_cer(ground_truth_str, emissions_str))
    except Exception:
        return 1.0


def _align_words_char_range(
    raw_words: List[str],
    matched_tokens: List[Dict[str, Any]],
) -> List[WordInterval]:
    """
    Maps ground-truth words to matched tokens using character-range overlap and
    fuses adjacent ground-truth words if they map to the exact same emission token span.
    """
    if not raw_words or not matched_tokens:
        return []

    num_words = len(raw_words)
    num_tokens = len(matched_tokens)

    # Calculate normalized character lengths for ground truth words
    word_norm_lens = [len(normalize_text_for_alignment(w)) or 1 for w in raw_words]
    total_word_chars = sum(word_norm_lens)

    # Calculate normalized character lengths for matched tokens
    token_norm_lens = [
        len(normalize_text_for_alignment(t["word"])) or 1 for t in matched_tokens
    ]
    total_token_chars = sum(token_norm_lens)

    # Determine token index assignment for each word based on character range position
    word_token_spans = []
    curr_char_accum = 0.0

    for w_i, raw_w in enumerate(raw_words):
        w_len = word_norm_lens[w_i]
        # Char ratio mid point or range for this word
        w_start_ratio = curr_char_accum / total_word_chars
        curr_char_accum += w_len
        w_end_ratio = curr_char_accum / total_word_chars

        # Map ratio to token indices
        t_start_idx = min(int(w_start_ratio * num_tokens), num_tokens - 1)
        t_end_idx = min(int(np.ceil(w_end_ratio * num_tokens)), num_tokens)
        if t_end_idx <= t_start_idx:
            t_end_idx = t_start_idx + 1

        word_token_spans.append((t_start_idx, t_end_idx))

    # Group / Fuse ground truth words that map to identical single token bounds
    fused_intervals: List[WordInterval] = []
    curr_fused_words: List[str] = []
    curr_span: Optional[tuple] = None
    curr_confs: List[float] = []

    for w_i, raw_w in enumerate(raw_words):
        span = word_token_spans[w_i]

        if curr_span is None:
            curr_span = span
            curr_fused_words = [raw_w]
            sub_tokens = matched_tokens[span[0] : span[1]]
            curr_confs = [t.get("confidence", 1.0) for t in sub_tokens]
        elif span == curr_span and (span[1] - span[0] == 1):
            # Same single token covers multiple consecutive words -> Fuse them
            curr_fused_words.append(raw_w)
        else:
            # Output previous fused word interval
            sub_tokens = matched_tokens[curr_span[0] : curr_span[1]]
            w_start = sub_tokens[0]["start_time"]
            w_end = sub_tokens[-1]["end_time"]
            avg_conf = float(np.mean(curr_confs)) if curr_confs else 1.0

            fused_intervals.append(
                WordInterval(
                    word=" ".join(curr_fused_words),
                    start_sec=w_start,
                    end_sec=w_end,
                    confidence=avg_conf,
                    flagged=avg_conf < 0.5,
                )
            )

            curr_span = span
            curr_fused_words = [raw_w]
            sub_tokens = matched_tokens[span[0] : span[1]]
            curr_confs = [t.get("confidence", 1.0) for t in sub_tokens]

    if curr_span and curr_fused_words:
        sub_tokens = matched_tokens[curr_span[0] : curr_span[1]]
        w_start = sub_tokens[0]["start_time"]
        w_end = sub_tokens[-1]["end_time"]
        avg_conf = float(np.mean(curr_confs)) if curr_confs else 1.0

        fused_intervals.append(
            WordInterval(
                word=" ".join(curr_fused_words),
                start_sec=w_start,
                end_sec=w_end,
                confidence=avg_conf,
                flagged=avg_conf < 0.5,
            )
        )

    return fused_intervals


def align_tokens_to_verses(
    token_emissions: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    audio_source: str = "",
) -> AlignmentResult:
    """
    Aligns raw CTC token emissions with global timestamps to ground-truth verses
    using Dynamic Time Warping (DTW) & character range sliding window search.

    Args:
        token_emissions: List of word dicts with start_time, end_time, and word text.
        verses: List of parsed ground-truth verse dicts from prepare_ground_truth.

    Returns:
        AlignmentResult data structure with populated verse & word interval timestamps.
    """
    aligned_verses = []

    if not token_emissions or not verses:
        for v in verses:
            aligned_verses.append(
                VerseInterval(
                    line_id=v.get("line_id", ""),
                    cherokee_syllabary=v.get("cherokee_syllabary", ""),
                    raw_phonetic=v.get("raw_phonetic", ""),
                    english=v.get("english", ""),
                    start_sec=0.0,
                    end_sec=0.0,
                    words=[],
                )
            )
        return AlignmentResult(audio_source=audio_source, verses=aligned_verses)

    num_tokens = len(token_emissions)

    # 1. Normalize verse texts & extract words
    normalized_verses = []
    for v in verses:
        norm_txt = v.get("normalized_text", "")
        if not norm_txt:
            norm_txt = normalize_text_for_alignment(v.get("raw_phonetic", ""))
        raw_words = [w for w in v.get("raw_phonetic", "").split() if w]
        if not raw_words:
            raw_words = norm_txt.split()
        normalized_verses.append(
            {
                "line_id": v.get("line_id", ""),
                "cherokee_syllabary": v.get("cherokee_syllabary", ""),
                "raw_phonetic": v.get("raw_phonetic", ""),
                "english": v.get("english", ""),
                "norm_text": norm_txt,
                "raw_words": raw_words,
            }
        )

    # 2. Build cost matrix for Dynamic Time Warping / Sliding Window
    token_idx = 0

    for v_idx, v in enumerate(normalized_verses):
        norm_txt = v["norm_text"]
        raw_words = v["raw_words"]
        num_words = len(raw_words)
        char_len = len(norm_txt)

        if not norm_txt or num_words == 0:
            aligned_verses.append(
                VerseInterval(
                    line_id=v["line_id"],
                    cherokee_syllabary=v["cherokee_syllabary"],
                    raw_phonetic=v["raw_phonetic"],
                    english=v["english"],
                    start_sec=0.0,
                    end_sec=0.0,
                    words=[],
                )
            )
            continue

        # 2D Sliding Window search (slide start skip and end length)
        best_cost = float("inf")
        best_start_idx = token_idx
        best_end_idx = token_idx

        # Allow skipping leading preamble tokens for verse 1 (and up to 5 tokens for subsequent verses)
        max_skip = min(num_tokens - token_idx, 30 if v_idx == 0 else 5)
        # Search length scaled by character length as well as word count
        max_search_len = min(num_tokens - token_idx, max(num_words * 3, 10))

        if max_search_len > 0:
            for skip in range(0, max_skip + 1):
                curr_start = token_idx + skip
                if curr_start >= num_tokens:
                    break

                for k in range(1, max_search_len + 1):
                    if curr_start + k > num_tokens:
                        break
                    candidate_tokens = token_emissions[curr_start : curr_start + k]
                    candidate_str = " ".join([t["word"] for t in candidate_tokens])
                    candidate_norm = normalize_text_for_alignment(candidate_str)
                    cost = compute_trigram_edit_cost(candidate_norm, norm_txt)

                    if cost < best_cost:
                        best_cost = cost
                        best_start_idx = curr_start
                        best_end_idx = curr_start + k

        matched_tokens = token_emissions[best_start_idx:best_end_idx]

        if matched_tokens:
            verse_start = matched_tokens[0]["start_time"]
            verse_end = matched_tokens[-1]["end_time"]

            # Map words using character-range alignment and fuse words mapping to single tokens
            word_intervals = _align_words_char_range(raw_words, matched_tokens)
            token_idx = best_end_idx
        else:
            # Fallback if no matching tokens available in stream
            prev_end = aligned_verses[-1].end_sec if aligned_verses else 0.0
            verse_start = prev_end
            verse_end = prev_end
            word_intervals = []

        aligned_verses.append(
            VerseInterval(
                line_id=v["line_id"],
                cherokee_syllabary=v["cherokee_syllabary"],
                raw_phonetic=v["raw_phonetic"],
                english=v["english"],
                start_sec=verse_start,
                end_sec=verse_end,
                words=word_intervals,
            )
        )

    return AlignmentResult(audio_source=audio_source, verses=aligned_verses)
