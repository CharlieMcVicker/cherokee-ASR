# -*- coding: utf-8 -*-
"""
aligner.py

ASR CTC emissions extraction & Trigram Sliding-Window DTW Alignment Engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from jiwer import cer as jiwer_cer
import numpy as np

from transcription.timestamping.prepare_ground_truth import normalize_text_for_alignment
from transcription.timestamping.audio_segmenter import segment_long_audio, AudioChunk


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

    Returns:
        AlignmentResult data structure.
    """
    return align_tokens_to_verses(
        token_emissions, verses, audio_source=audio_source, reconcile=reconcile
    )


def align_audio_segment(
    audio_input: Any,
    verses: List[Dict[str, Any]],
    model_or_fn: Optional[Any] = None,
    processor: Optional[Any] = None,
    audio_source: str = "",
    skip_vad: bool = False,
    token_emissions: Optional[List[Dict[str, Any]]] = None,
    reconcile: bool = False,
) -> AlignmentResult:
    """
    Exposes audio/emissions alignment for in-memory execution.

    Args:
        audio_input: Audio file path, AudioSegment, or raw audio waveform sample array.
        verses: List of ground-truth verse/segment dicts.
        model_or_fn: PyTorch Wav2Vec2 model instance or callable taking samples array and returning word emissions list,
                     or None if token_emissions are provided directly.
        processor: Wav2Vec2Processor instance if model_or_fn is a PyTorch Wav2Vec2 model.
        audio_source: Optional label or path string for output metadata.
        skip_vad: If True, bypasses VAD audio pre-segmentation when operating on pre-cut audio clips.
        token_emissions: Pre-computed token emissions list (dicts with 'word', 'start_time', 'end_time', 'confidence').
        reconcile: If True, performs phonological syllabary/ASR reconciliation on aligned words.

    Returns:
        AlignmentResult data structure.
    """
    if token_emissions is not None:
        return align_emissions_to_text(
            token_emissions, verses, audio_source=audio_source, reconcile=reconcile
        )

    if skip_vad:
        # Bypass VAD segmentation - handle audio as a single chunk
        from pydub import AudioSegment

        if isinstance(audio_input, str):
            audio_seg = AudioSegment.from_file(audio_input)
        elif isinstance(audio_input, AudioSegment):
            audio_seg = audio_input
        else:
            # If raw numpy array or torch tensor audio samples passed in memory
            # caller can pass token_emissions or AudioSegment
            audio_seg = audio_input

        if isinstance(audio_seg, AudioSegment):
            chunks = [
                AudioChunk(
                    chunk_index=0,
                    audio=audio_seg,
                    start_sec=0.0,
                    end_sec=round(len(audio_seg) / 1000.0, 3),
                )
            ]
        else:
            chunks = []
    else:
        chunks = segment_long_audio(audio_input)

    if token_emissions is None:
        if model_or_fn is None:
            raise ValueError(
                "Either token_emissions or model_or_fn must be provided to align_audio_segment."
            )

        extracted_tokens = []
        import torch

        for c in chunks:
            if callable(model_or_fn) and processor is None:
                # Custom inference callback function: fn(samples, sample_rate) -> list of word dicts
                samples = np.array(c.audio.get_array_of_samples(), dtype=np.float32)
                if c.audio.channels > 1:
                    samples = samples.reshape((-1, c.audio.channels)).mean(axis=1)
                max_val = float(1 << (8 * c.audio.sample_width - 1))
                samples = samples / max_val
                chunk_words: Any = model_or_fn(samples, c.audio.frame_rate)
            else:
                # PyTorch model + processor
                from transcription.inference.infer import calculate_word_confidences

                samples = np.array(c.audio.get_array_of_samples(), dtype=np.float32)
                if c.audio.channels > 1:
                    samples = samples.reshape((-1, c.audio.channels)).mean(axis=1)
                max_val = float(1 << (8 * c.audio.sample_width - 1))
                samples = samples / max_val

                model_obj: Any = model_or_fn
                proc_obj: Any = processor
                if proc_obj is None or model_obj is None:
                    raise ValueError(
                        "Both model_or_fn and processor must be provided for PyTorch inference."
                    )

                device = (
                    next(model_obj.parameters()).device
                    if hasattr(model_obj, "parameters")
                    else "cpu"
                )
                input_values = proc_obj(
                    samples, sampling_rate=c.audio.frame_rate, return_tensors="pt"
                ).input_values.to(device)
                with torch.no_grad():
                    logits = model_obj(input_values).logits[0]

                pred_ids = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                chunk_words = calculate_word_confidences(probs, pred_ids, proc_obj)

            for w_info in chunk_words:
                extracted_tokens.append(
                    {
                        "word": w_info["word"],
                        "start_time": round(c.start_sec + w_info["start_time"], 3),
                        "end_time": round(c.start_sec + w_info["end_time"], 3),
                        "confidence": w_info.get("confidence", 1.0),
                    }
                )
        token_emissions = extracted_tokens

    return align_emissions_to_text(
        token_emissions, verses, audio_source=audio_source, reconcile=reconcile
    )


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
    raw_syllabary_words: Optional[List[str]] = None,
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
    MAX_FUSE_GT = 4  # maximum consecutive GT words allowed to fuse
    MAX_FUSE_ASR = 3  # maximum consecutive ASR tokens allowed to fuse

    dp = np.full((N + 1, M + 1), fill_value=1e9, dtype=np.float32)
    parent: List[List[Optional[Tuple[int, int, str]]]] = [
        [None for _ in range(M + 1)] for _ in range(N + 1)
    ]

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

            # Option 3: Match M ASR tokens against K GT words
            for m_len in range(1, MAX_FUSE_ASR + 1):
                if j + m_len <= M:
                    asr_concat = "".join(norm_tokens[j : j + m_len])
                    for k in range(1, MAX_FUSE_GT + 1):
                        if i + k <= N:
                            gt_concat = " ".join(norm_words[i : i + k])
                            edit_cost = compute_trigram_edit_cost(asr_concat, gt_concat)
                            # Penalty per additional fused element (0.15) to prefer 1-to-1 matches
                            # unless fusion provides a clear improvement in edit distance.
                            penalty = 0.15 * (k - 1) + 0.15 * (m_len - 1)
                            cost = current_cost + edit_cost + penalty

                            if cost < dp[i + k, j + m_len]:
                                dp[i + k][j + m_len] = cost
                                parent[i + k][j + m_len] = (
                                    i,
                                    j,
                                    f"match_fuse_{k}_{m_len}",
                                )

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
            syll_w = (
                raw_syllabary_words[pi]
                if (raw_syllabary_words and pi < len(raw_syllabary_words))
                else ""
            )
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
                    cherokee_syllabary=syll_w,
                )
            )
        elif action_type == "gap_token":
            continue
        elif action_type.startswith("match_fuse_"):
            parts = action_type.split("_")
            k = int(parts[2])
            m_len = int(parts[3]) if len(parts) > 3 else 1
            fused_gt_text = " ".join(raw_words[pi : pi + k])
            syll_w = (
                " ".join(raw_syllabary_words[pi : pi + k])
                if (raw_syllabary_words and pi + k <= len(raw_syllabary_words))
                else ""
            )

            first_tok = matched_tokens[pj]
            last_tok = matched_tokens[pj + m_len - 1]

            confidences = [
                t.get("confidence", 1.0) for t in matched_tokens[pj : pj + m_len]
            ]
            avg_conf = float(np.mean(confidences)) if confidences else 1.0

            tok_words = " ".join([t["word"] for t in matched_tokens[pj : pj + m_len]])

            fused_intervals.append(
                WordInterval(
                    word=fused_gt_text,
                    start_sec=first_tok["start_time"],
                    end_sec=last_tok["end_time"],
                    confidence=avg_conf,
                    flagged=avg_conf < 0.5,
                    cherokee_syllabary=syll_w,
                    emitted_word=tok_words,
                )
            )

    return fused_intervals


def align_tokens_to_verses(
    token_emissions: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    audio_source: str = "",
    reconcile: bool = False,
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
                    tok_words = [t["word"] for t in candidate_tokens]
                    candidate_norm_spaced = normalize_text_for_alignment(
                        " ".join(tok_words)
                    )
                    candidate_norm_concat = normalize_text_for_alignment(
                        "".join(tok_words)
                    )

                    cost_spaced = compute_trigram_edit_cost(
                        candidate_norm_spaced, norm_txt
                    )
                    cost_concat = compute_trigram_edit_cost(
                        candidate_norm_concat, norm_txt
                    )
                    cost = min(cost_spaced, cost_concat)

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

            raw_syllabary_words = [
                w for w in v.get("cherokee_syllabary", "").split() if w
            ]
            # Map words using character-range alignment and fuse words mapping to single tokens
            word_intervals = _align_words_char_range(
                raw_words, matched_tokens, raw_syllabary_words=raw_syllabary_words
            )

            if reconcile and word_intervals:
                from transcription.syllabary_enrichment import (
                    align_character_syllable,
                    reconcile_phonetics,
                )

                for w_int in word_intervals:
                    target_emitted = w_int.emitted_word or emitted_text
                    if w_int.cherokee_syllabary and target_emitted:
                        align_pairs = align_character_syllable(
                            w_int.cherokee_syllabary, target_emitted
                        )
                        base_trans = "".join([pair[0] for pair in align_pairs])
                        try:
                            w_int.reconciled_word = reconcile_phonetics(
                                syllabary_text=w_int.cherokee_syllabary,
                                base_transliteration=base_trans,
                                emitted_text=target_emitted,
                                aligned_pairs=align_pairs,
                            )
                        except Exception:
                            w_int.reconciled_word = w_int.word
                    else:
                        w_int.reconciled_word = w_int.word

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
