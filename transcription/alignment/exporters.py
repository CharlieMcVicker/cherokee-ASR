# -*- coding: utf-8 -*-
"""
exporters.py

Pure export functions for saving AlignmentOutput into Praat .TextGrid, alignment_manifest.json,
and alignment_debug.json formats.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from transcription.alignment.models import AlignmentOutput, WordInterval


def _build_contiguous_intervals(
    raw_intervals: List[Dict[str, Any]], total_end: float
) -> List[Tuple[float, float, str]]:
    """Builds a contiguous sequence of non-overlapping intervals spanning 0.0 to total_end."""
    contiguous: List[Tuple[float, float, str]] = []
    curr_time = 0.0

    valid_intervals = [i for i in raw_intervals if i["end_sec"] > i["start_sec"]]
    valid_intervals.sort(key=lambda x: x["start_sec"])

    for item in valid_intervals:
        start_sec = max(0.0, item["start_sec"])
        end_sec = max(start_sec, item["end_sec"])

        if start_sec > curr_time + 1e-4:
            contiguous.append((curr_time, start_sec, ""))
            curr_time = start_sec
        elif start_sec < curr_time:
            start_sec = curr_time
            if end_sec <= start_sec:
                continue

        contiguous.append((start_sec, end_sec, item["text"]))
        curr_time = end_sec

    if curr_time < total_end:
        contiguous.append((curr_time, total_end, ""))

    if not contiguous:
        contiguous.append((0.0, max(1.0, total_end), ""))

    return contiguous


def _build_padded_word_intervals(
    word_raw: List[Dict[str, Any]], total_end: float, pad_sec: float = 0.10
) -> List[Tuple[float, float, str]]:
    """Builds padded word intervals with temporal fusion of overlapping boundaries."""
    valid_words = [w for w in word_raw if w["end_sec"] > w["start_sec"]]
    valid_words.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

    if not valid_words:
        return _build_contiguous_intervals([], total_end)

    fused: List[Dict[str, Any]] = []
    curr_start = valid_words[0]["start_sec"]
    curr_end = valid_words[0]["end_sec"] + pad_sec
    curr_texts = [valid_words[0]["text"]]

    for w in valid_words[1:]:
        next_start = w["start_sec"]
        next_end = w["end_sec"] + pad_sec
        next_text = w["text"]

        if curr_end > next_start:
            curr_end = max(curr_end, next_end)
            curr_texts.append(next_text)
        else:
            fused.append(
                {
                    "start_sec": curr_start,
                    "end_sec": curr_end,
                    "text": " ".join(curr_texts),
                }
            )
            curr_start = next_start
            curr_end = next_end
            curr_texts = [next_text]

    fused.append(
        {
            "start_sec": curr_start,
            "end_sec": curr_end,
            "text": " ".join(curr_texts),
        }
    )

    return _build_contiguous_intervals(fused, total_end)


def _format_tier(
    tier_idx: int,
    tier_name: str,
    intervals: List[Tuple[float, float, str]],
    max_t: float,
) -> List[str]:
    """Formats an IntervalTier block for Praat TextGrid."""
    t_lines = [
        f"    item [{tier_idx}]:",
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        "        xmin = 0",
        f"        xmax = {max_t:.3f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for idx, (xmin, xmax, label) in enumerate(intervals, 1):
        t_lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin:.3f}",
                f"            xmax = {xmax:.3f}",
                f'            text = "{label}"',
            ]
        )
    return t_lines


def export_textgrid(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "alignment.TextGrid",
    pad_sec: float = 0.10,
    source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_word_tiers: Optional[Mapping[str, Sequence[WordInterval]]] = None,
) -> str:
    """
    Generates Praat TextGrid with Chunks, Words, Padded Words, custom additional word tiers, and optional Emissions tiers.

    Args:
        alignment: AlignmentOutput object.
        output_dir: Directory path where TextGrid should be saved.
        filename: Name of the TextGrid output file.
        pad_sec: Padding seconds added to words for the Padded Words tier.
        source_metadata: Optional chunk metadata dictionary.
        additional_word_tiers: Optional mapping of tier name to Sequence of WordIntervals.

    Returns:
        Absolute or relative path to the generated TextGrid file.
    """
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), filename)

    total_end = 0.0
    if alignment.aligned_chunks:
        total_end = max(c.end_sec for c in alignment.aligned_chunks)
    if alignment.raw_tokens:
        total_end = max(total_end, max(t.end_sec for t in alignment.raw_tokens))
    if additional_word_tiers:
        for word_seq in additional_word_tiers.values():
            if word_seq:
                total_end = max(total_end, max(w.end_sec for w in word_seq))
    if total_end <= 0.0:
        total_end = 1.0

    meta_lookup = source_metadata or {}

    # Tier 1: Chunks
    chunk_raw = []
    for c in alignment.aligned_chunks:
        text_val = meta_lookup.get(c.chunk_id, {}).get("text", "")
        chunk_label = f"{c.chunk_id}: {text_val}" if text_val else c.chunk_id
        chunk_raw.append(
            {
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "text": chunk_label,
            }
        )
    chunk_intervals = _build_contiguous_intervals(chunk_raw, total_end)

    # Tier 2: Words
    word_raw = []
    for c in alignment.aligned_chunks:
        for w in c.words:
            word_raw.append(
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            )

    word_intervals = _build_contiguous_intervals(word_raw, total_end)
    padded_word_intervals = _build_padded_word_intervals(
        word_raw, total_end, pad_sec=pad_sec
    )

    # Build additional custom tiers
    extra_tiers_formatted: List[Tuple[str, List[Tuple[float, float, str]]]] = []
    if additional_word_tiers:
        for tier_name, word_seq in additional_word_tiers.items():
            tier_raw = [
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
                for w in word_seq
            ]
            tier_intervals = _build_contiguous_intervals(tier_raw, total_end)
            extra_tiers_formatted.append((tier_name, tier_intervals))

    raw_token_items = [
        {
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "text": t.word,
        }
        for t in (alignment.raw_tokens or [])
    ]
    has_raw_tokens = bool(raw_token_items)

    tier_count = 3 + len(extra_tiers_formatted)
    if has_raw_tokens:
        tier_count += 1

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {total_end:.3f}",
        "tiers? <exists>",
        f"size = {tier_count}",
        "item []:",
    ]

    tier_idx = 1
    lines.extend(_format_tier(tier_idx, "Chunks", chunk_intervals, total_end))
    tier_idx += 1

    lines.extend(_format_tier(tier_idx, "Words", word_intervals, total_end))
    tier_idx += 1

    lines.extend(
        _format_tier(tier_idx, "Padded Words", padded_word_intervals, total_end)
    )
    tier_idx += 1

    for tier_name, intervals in extra_tiers_formatted:
        lines.extend(_format_tier(tier_idx, tier_name, intervals, total_end))
        tier_idx += 1

    if has_raw_tokens:
        raw_token_intervals = _build_contiguous_intervals(raw_token_items, total_end)
        lines.extend(
            _format_tier(tier_idx, "Raw ASR Emissions", raw_token_intervals, total_end)
        )
        tier_idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def _serialize_word_interval(w: WordInterval) -> Dict[str, Any]:
    """Helper to convert WordInterval into JSON serializable dict."""
    w_dict: Dict[str, Any] = {
        "word": w.word,
        "start": w.start_sec,
        "end": w.end_sec,
        "confidence": w.confidence,
        "flagged": w.flagged,
    }
    if w.emitted_word:
        w_dict["emitted_word"] = w.emitted_word
    return w_dict


def export_manifest(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "alignment_manifest.json",
    source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_word_tiers: Optional[Mapping[str, Sequence[WordInterval]]] = None,
) -> str:
    """
    Generates manifest JSON matching project schema, including optional additional word tiers.

    Args:
        alignment: AlignmentOutput object.
        output_dir: Directory path where manifest JSON should be saved.
        filename: Name of the manifest JSON output file.
        source_metadata: Optional chunk metadata dictionary.
        additional_word_tiers: Optional mapping of tier name to Sequence of WordIntervals.

    Returns:
        Path to the generated JSON manifest file.
    """
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), filename)

    manifest_lines = []
    meta_lookup = source_metadata or {}

    tier_chunk_words: Dict[str, List[List[WordInterval]]] = {}
    if additional_word_tiers:
        total_chunk_words = sum(len(c.words) for c in alignment.aligned_chunks)
        for tier_name, tier_words in additional_word_tiers.items():
            tier_word_list = list(tier_words)
            if len(tier_word_list) == total_chunk_words:
                cur = 0
                chunk_lists: List[List[WordInterval]] = []
                for c in alignment.aligned_chunks:
                    chunk_len = len(c.words)
                    chunk_lists.append(tier_word_list[cur : cur + chunk_len])
                    cur += chunk_len
                tier_chunk_words[tier_name] = chunk_lists
            else:
                chunk_lists = []
                for c in alignment.aligned_chunks:
                    matched = [
                        w
                        for w in tier_word_list
                        if (
                            w.start_sec >= c.start_sec - 0.05
                            and w.end_sec <= c.end_sec + 0.05
                        )
                        or (w.start_sec < c.end_sec and w.end_sec > c.start_sec)
                    ]
                    chunk_lists.append(matched)
                tier_chunk_words[tier_name] = chunk_lists

    for c_idx, c in enumerate(alignment.aligned_chunks):
        word_objs = []
        for w_idx, w in enumerate(c.words):
            w_dict = _serialize_word_interval(w)
            rec_tier = tier_chunk_words.get("Reconciled Words") or tier_chunk_words.get(
                "Reconciled Transcriptions"
            )
            if rec_tier:
                rec_words_for_chunk = rec_tier[c_idx]
                if w_idx < len(rec_words_for_chunk):
                    w_dict["reconciled_word"] = rec_words_for_chunk[w_idx].word
            word_objs.append(w_dict)

        chunk_meta = meta_lookup.get(c.chunk_id, {})
        line_dict: Dict[str, Any] = {
            "line_id": c.chunk_id,
            "cherokee_syllabary": chunk_meta.get(
                "cherokee_syllabary", chunk_meta.get("cherokee", "")
            ),
            "text": chunk_meta.get("text", ""),
            "english": chunk_meta.get("english", ""),
            "start": c.start_sec,
            "end": c.end_sec,
            "cer": c.distance_score,
            "emitted_text": c.emitted_text,
            "words": word_objs,
        }

        if additional_word_tiers:
            chunk_additional_tiers: Dict[str, List[Dict[str, Any]]] = {}
            for tier_name in additional_word_tiers:
                tier_words_for_chunk = tier_chunk_words[tier_name][c_idx]
                chunk_additional_tiers[tier_name] = [
                    _serialize_word_interval(tw) for tw in tier_words_for_chunk
                ]

            line_dict["additional_word_tiers"] = chunk_additional_tiers
            rec_chunk = chunk_additional_tiers.get(
                "Reconciled Words"
            ) or chunk_additional_tiers.get("Reconciled Transcriptions")
            if rec_chunk is not None:
                line_dict["reconciled_words"] = rec_chunk

        manifest_lines.append(line_dict)

    metrics_dict: Dict[str, Any] = {}
    if alignment.metrics:
        metrics_dict = {
            "total_chunks": alignment.metrics.total_chunks,
            "matched_chunks": alignment.metrics.matched_chunks,
            "match_ratio": alignment.metrics.match_ratio,
            "mean_distance_score": alignment.metrics.mean_distance_score,
            "total_ground_truth_chars": alignment.metrics.total_ground_truth_chars,
            "total_emitted_chars": alignment.metrics.total_emitted_chars,
        }

    data: Dict[str, Any] = {
        "audio_source": alignment.source_id,
        "metrics": metrics_dict,
        "lines": manifest_lines,
    }

    if additional_word_tiers:
        top_additional_tiers: Dict[str, List[Dict[str, Any]]] = {}
        for tier_name, tier_words in additional_word_tiers.items():
            top_additional_tiers[tier_name] = [
                _serialize_word_interval(tw) for tw in tier_words
            ]
        data["additional_word_tiers"] = top_additional_tiers
        rec_top = top_additional_tiers.get(
            "Reconciled Words"
        ) or top_additional_tiers.get("Reconciled Transcriptions")
        if rec_top is not None:
            data["reconciled_words"] = rec_top

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def export_debug_json(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "alignment_debug.json",
) -> str:
    """
    Writes raw tokens, counts, and metrics JSON into output_dir.

    Args:
        alignment: AlignmentOutput object.
        output_dir: Directory path where debug JSON should be saved.
        filename: Name of the debug JSON output file.

    Returns:
        Path to the generated debug JSON file.
    """
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), filename)

    raw_toks = [
        {
            "word": t.word,
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "confidence": t.confidence,
        }
        for t in (alignment.raw_tokens or [])
    ]

    metrics_dict: Dict[str, Any] = {}
    if alignment.metrics:
        metrics_dict = {
            "total_chunks": alignment.metrics.total_chunks,
            "matched_chunks": alignment.metrics.matched_chunks,
            "match_ratio": alignment.metrics.match_ratio,
            "mean_distance_score": alignment.metrics.mean_distance_score,
            "total_ground_truth_chars": alignment.metrics.total_ground_truth_chars,
            "total_emitted_chars": alignment.metrics.total_emitted_chars,
        }

    debug_data = {
        "audio_source": alignment.source_id,
        "raw_tokens": raw_toks,
        "aligned_chunks_count": len(alignment.aligned_chunks),
        "metrics": metrics_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    return output_path
