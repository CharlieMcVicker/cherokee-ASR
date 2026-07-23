# -*- coding: utf-8 -*-
"""
exporter.py

Praat TextGrid and JSON alignment_manifest.json export module.
"""

import json
import os
from typing import Dict, Any
from transcription.timestamping.aligner import AlignmentResult


def _build_contiguous_intervals(raw_intervals, total_end: float):
    """
    Constructs a fully contiguous list of (xmin, xmax, text) intervals for Praat IntervalTier.
    Inserts empty text "" intervals for any gaps, ensuring xmin[i] == xmax[i-1], starting at 0 and ending at total_end.
    """
    contiguous = []
    curr_time = 0.0

    # Sort intervals by start_sec
    valid_intervals = [i for i in raw_intervals if i["end_sec"] > i["start_sec"]]
    valid_intervals.sort(key=lambda x: x["start_sec"])

    for item in valid_intervals:
        start_sec = max(0.0, item["start_sec"])
        end_sec = max(start_sec, item["end_sec"])

        if start_sec > curr_time + 1e-4:
            # Fill gap with empty interval
            contiguous.append((curr_time, start_sec, ""))
            curr_time = start_sec
        elif start_sec < curr_time:
            # Prevent overlap / backwards step
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


def export_praat_textgrid(alignment: AlignmentResult, output_path: str) -> None:
    """
    Generates a valid 3-tier Praat .TextGrid file:
    - Tier 1: Verses (IntervalTier)
    - Tier 2: Words (IntervalTier - Ground Truth aligned words)
    - Tier 3: Raw ASR Emissions (IntervalTier - Original model word tokens)
    Fills gaps with unannotated empty intervals to meet Praat's strict contiguous IntervalTier specification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine total audio end time
    total_end = 0.0
    if alignment.verses:
        total_end = max(v.end_sec for v in alignment.verses)
    if alignment.raw_tokens:
        total_end = max(
            total_end, max(t.get("end_time", 0.0) for t in alignment.raw_tokens)
        )
    if total_end <= 0.0:
        total_end = 1.0

    # Build Tier 1: Verses
    verse_raw = [
        {
            "start_sec": v.start_sec,
            "end_sec": v.end_sec,
            "text": f"{v.line_id}: {v.raw_phonetic}",
        }
        for v in alignment.verses
    ]
    verse_intervals = _build_contiguous_intervals(verse_raw, total_end)

    # Build Tier 2: Words (Ground Truth mapped)
    word_raw = []
    for v in alignment.verses:
        for w in v.words:
            word_raw.append(
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            )
    word_intervals = _build_contiguous_intervals(word_raw, total_end)

    # Build Tier 3: Raw ASR Emissions (Original model output tokens)
    raw_token_items = [
        {
            "start_sec": t.get("start_time", 0.0),
            "end_sec": t.get("end_time", 0.0),
            "text": t.get("word", ""),
        }
        for t in (alignment.raw_tokens or [])
    ]
    raw_token_intervals = _build_contiguous_intervals(raw_token_items, total_end)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {total_end:.3f}",
        "tiers? <exists>",
        "size = 3",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "Verses"',
        "        xmin = 0",
        f"        xmax = {total_end:.3f}",
        f"        intervals: size = {len(verse_intervals)}",
    ]

    for idx, (xmin, xmax, label) in enumerate(verse_intervals, 1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin:.3f}",
                f"            xmax = {xmax:.3f}",
                f'            text = "{label}"',
            ]
        )

    lines.extend(
        [
            "    item [2]:",
            '        class = "IntervalTier"',
            '        name = "Words"',
            "        xmin = 0",
            f"        xmax = {total_end:.3f}",
            f"        intervals: size = {len(word_intervals)}",
        ]
    )

    for idx, (xmin, xmax, label) in enumerate(word_intervals, 1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin:.3f}",
                f"            xmax = {xmax:.3f}",
                f'            text = "{label}"',
            ]
        )

    lines.extend(
        [
            "    item [3]:",
            '        class = "IntervalTier"',
            '        name = "Raw ASR Emissions"',
            "        xmin = 0",
            f"        xmax = {total_end:.3f}",
            f"        intervals: size = {len(raw_token_intervals)}",
        ]
    )

    for idx, (xmin, xmax, label) in enumerate(raw_token_intervals, 1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin:.3f}",
                f"            xmax = {xmax:.3f}",
                f'            text = "{label}"',
            ]
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_alignment_manifest(alignment: AlignmentResult, output_path: str) -> None:
    """
    Exports alignment_manifest.json following ground-truth app schema including metrics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    manifest_lines = []
    for v in alignment.verses:
        word_objs = [
            {
                "word": w.word,
                "start": w.start_sec,
                "end": w.end_sec,
                "confidence": w.confidence,
                "flagged": w.flagged,
            }
            for w in v.words
        ]
        manifest_lines.append(
            {
                "line_id": v.line_id,
                "cherokee_syllabary": v.cherokee_syllabary,
                "text": v.raw_phonetic,
                "english": v.english,
                "start": v.start_sec,
                "end": v.end_sec,
                "cer": v.cer,
                "emitted_text": v.emitted_text,
                "words": word_objs,
            }
        )

    metrics_dict = {}
    if alignment.metrics:
        metrics_dict = {
            "total_verses": alignment.metrics.total_verses,
            "matched_verses": alignment.metrics.matched_verses,
            "matched_verse_ratio": alignment.metrics.matched_verse_ratio,
            "overall_cer": alignment.metrics.overall_cer,
            "mean_verse_cer": alignment.metrics.mean_verse_cer,
            "total_ground_truth_chars": alignment.metrics.total_ground_truth_chars,
            "total_emitted_chars": alignment.metrics.total_emitted_chars,
        }

    data = {
        "audio_source": alignment.audio_source,
        "metrics": metrics_dict,
        "lines": manifest_lines,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
