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
    if not emissions_str or not ground_truth_str:
        return 1.0
    try:
        return float(jiwer_cer(ground_truth_str, emissions_str))
    except Exception:
        return 1.0


def compute_alignment_metrics(alignment_result: AlignmentResult) -> AlignmentMetrics:
    """
    Computes segment alignment quality metrics exclusively across matched verses.
    """
    verses = alignment_result.verses
    total_verses = len(verses)
    if total_verses == 0:
        return AlignmentMetrics(
            total_verses=0,
            matched_verses=0,
            matched_verse_ratio=0.0,
            overall_cer=1.0,
            mean_verse_cer=1.0,
            total_ground_truth_chars=0,
            total_emitted_chars=0,
        )

    matched_verse_list = [v for v in verses if v.words and (v.end_sec > v.start_sec)]
    matched_verses_count = len(matched_verse_list)
    matched_ratio = round(matched_verses_count / total_verses, 4)

    verse_cers = [v.cer for v in matched_verse_list]
    mean_verse_cer = round(float(np.mean(verse_cers)), 4) if verse_cers else 1.0

    # CER exclusively across matched verses
    matched_gt = []
    matched_emitted = []
    for v in matched_verse_list:
        gt_norm = normalize_text_for_alignment(v.raw_phonetic)
        em_norm = normalize_text_for_alignment(v.emitted_text)
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
) -> List[WordInterval]:
    """
    Aligns ground-truth words to matched emission tokens using Needleman-Wunsch DP string edit distance.
    Evaluates GT word fusion (1 token mapping to N GT words) as valid DP transitions.
    Preserves exact ASR emission token start_time and end_time boundaries for matched tokens.
    """
    if not raw_words or not matched_tokens:
        return []

    N = len(raw_words)
    M = len(matched_tokens)

    # Pre-normalize raw words and tokens
    norm_words = [normalize_text_for_alignment(w) for w in raw_words]
    norm_tokens = [normalize_text_for_alignment(t["word"]) for t in matched_tokens]

    # DP state: dp[i, j] = (min_cost, parent_i, parent_j)
    GAP_COST = 0.8
    MAX_FUSE_GT = 4  # maximum consecutive GT words allowed to fuse onto 1 ASR token

    dp = np.full((N + 1, M + 1), fill_value=1e9, dtype=np.float32)
    parent = [[None for _ in range(M + 1)] for _ in range(N + 1)]

    dp[0, 0] = 0.0

    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i, j] >= 1e8:
                continue

            current_cost = dp[i, j]

            # Option 1: Unaligned GT word (Deletion gap)
            if i + 1 <= N:
                cost = current_cost + GAP_COST
                if cost < dp[i + 1, j]:
                    dp[i + 1, j] = cost
                    parent[i + 1][j] = (i, j, "gap_gt")

            # Option 2: Unaligned ASR token (Insertion gap)
            if j + 1 <= M:
                cost = current_cost + GAP_COST
                if cost < dp[i, j + 1]:
                    dp[i, j + 1] = cost
                    parent[i][j + 1] = (i, j, "gap_token")

            # Option 3: Match 1 token against K GT words (K = 1..MAX_FUSE_GT)
            if j + 1 <= M:
                t_str = norm_tokens[j]
                for k in range(1, MAX_FUSE_GT + 1):
                    if i + k <= N:
                        gt_concat = " ".join(norm_words[i : i + k])
                        edit_cost = compute_trigram_edit_cost(t_str, gt_concat)
                        # Add slight penalty per fused word to prefer 1-to-1 matches when cost is equal
                        cost = current_cost + edit_cost + (0.05 * (k - 1))

                        if cost < dp[i + k, j + 1]:
                            dp[i + k][j + 1] = cost
                            parent[i + k][j + 1] = (i, j, f"match_fuse_{k}")

    # Backtrack optimal DP path
    i, j = N, M
    actions = []

    while i > 0 or j > 0:
        p = parent[i][j]
        if p is None:
            break
        pi, pj, action_type = p
        actions.append((pi, pj, i, j, action_type))
        i, j = pi, pj

    actions.reverse()

    # Convert DP actions into WordInterval list
    fused_intervals: List[WordInterval] = []

    for pi, pj, i, j, action_type in actions:
        if action_type == "gap_gt":
            # Unaligned GT word: Interpolate timestamp
            raw_w = raw_words[pi]
            prev_end = (
                fused_intervals[-1].end_sec
                if fused_intervals
                else matched_tokens[0]["start_time"]
            )
            fused_intervals.append(
                WordInterval(
                    word=raw_w,
                    start_sec=prev_end,
                    end_sec=prev_end,
                    confidence=0.0,
                    flagged=True,
                )
            )
        elif action_type == "gap_token":
            continue
        elif action_type.startswith("match_fuse_"):
            k = int(action_type.split("_")[-1])
            fused_gt_text = " ".join(raw_words[pi : pi + k])
            matched_tok = matched_tokens[pj]

            fused_intervals.append(
                WordInterval(
                    word=fused_gt_text,
                    start_sec=matched_tok["start_time"],
                    end_sec=matched_tok["end_time"],
                    confidence=matched_tok.get("confidence", 1.0),
                    flagged=matched_tok.get("confidence", 1.0) < 0.5,
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
        AlignmentResult data structure with populated verse & word interval timestamps and CER metrics.
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
                    cer=1.0,
                    emitted_text="",
                )
            )
        res = AlignmentResult(audio_source=audio_source, verses=aligned_verses)
        compute_alignment_metrics(res)
        return res

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
                    cer=1.0,
                    emitted_text="",
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
            emitted_text = " ".join([t["word"] for t in matched_tokens])
            emitted_norm = normalize_text_for_alignment(emitted_text)
            verse_cer = compute_trigram_edit_cost(emitted_norm, norm_txt)

            # Map words using character-range alignment and fuse words mapping to single tokens
            word_intervals = _align_words_char_range(raw_words, matched_tokens)
            token_idx = best_end_idx
        else:
            # Fallback if no matching tokens available in stream
            prev_end = aligned_verses[-1].end_sec if aligned_verses else 0.0
            verse_start = prev_end
            verse_end = prev_end
            word_intervals = []
            emitted_text = ""
            verse_cer = 1.0

        aligned_verses.append(
            VerseInterval(
                line_id=v["line_id"],
                cherokee_syllabary=v["cherokee_syllabary"],
                raw_phonetic=v["raw_phonetic"],
                english=v["english"],
                start_sec=verse_start,
                end_sec=verse_end,
                words=word_intervals,
                cer=round(verse_cer, 4),
                emitted_text=emitted_text,
            )
        )

    res = AlignmentResult(
        audio_source=audio_source, verses=aligned_verses, raw_tokens=token_emissions
    )
    compute_alignment_metrics(res)
    return res
