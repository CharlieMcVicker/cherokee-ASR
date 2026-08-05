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


def _build_padded_word_intervals(word_raw, total_end: float, pad_sec: float = 0.10):
    """
    Pads word boundaries by adding pad_sec (10ms) to end times.
    If padded end overlaps with subsequent word start time, fuses groundtruth words
    and emits a single combined segment in the grid.
    """
    valid_words = [w for w in word_raw if w["end_sec"] > w["start_sec"]]
    valid_words.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

    if not valid_words:
        return _build_contiguous_intervals([], total_end)

    fused = []
    curr_start = valid_words[0]["start_sec"]
    curr_end = valid_words[0]["end_sec"] + pad_sec
    curr_texts = [valid_words[0]["text"]]

    for w in valid_words[1:]:
        next_start = w["start_sec"]
        next_end = w["end_sec"] + pad_sec
        next_text = w["text"]

        if curr_end > next_start:
            # Overlap occurs, fuse words and extend end time if needed
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
        {"start_sec": curr_start, "end_sec": curr_end, "text": " ".join(curr_texts)}
    )

    return _build_contiguous_intervals(fused, total_end)


def export_praat_textgrid(alignment: AlignmentResult, output_path: str) -> None:
    """
    Generates a valid 4-tier Praat .TextGrid file:
    - Tier 1: Verses (IntervalTier)
    - Tier 2: Words (IntervalTier - Ground Truth aligned words)
    - Tier 3: Padded Words (IntervalTier - Ground Truth aligned words + 10ms padding / fusion)
    - Tier 4: Raw ASR Emissions (IntervalTier - Original model word tokens)
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

    # Build Tier 3: Padded Words (GT mapped + 10ms pad + fused overlaps)
    padded_word_intervals = _build_padded_word_intervals(
        word_raw, total_end, pad_sec=0.10
    )

    # Check if reconciled words exist
    reconciled_raw = []
    has_reconciled = False
    for v in alignment.verses:
        for w in v.words:
            rec_text = getattr(w, "reconciled_word", "") or w.word
            if getattr(w, "reconciled_word", ""):
                has_reconciled = True
            reconciled_raw.append(
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": rec_text}
            )

    tier_count = 5 if has_reconciled else 4

    # Build Tier 4: Raw ASR Emissions (Original model output tokens)
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
        f"size = {tier_count}",
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
            '        name = "Padded Words"',
            "        xmin = 0",
            f"        xmax = {total_end:.3f}",
            f"        intervals: size = {len(padded_word_intervals)}",
        ]
    )

    for idx, (xmin, xmax, label) in enumerate(padded_word_intervals, 1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin:.3f}",
                f"            xmax = {xmax:.3f}",
                f'            text = "{label}"',
            ]
        )

    curr_item = 4
    if has_reconciled:
        reconciled_intervals = _build_contiguous_intervals(reconciled_raw, total_end)
        lines.extend(
            [
                f"    item [{curr_item}]:",
                '        class = "IntervalTier"',
                '        name = "Reconciled Words"',
                "        xmin = 0",
                f"        xmax = {total_end:.3f}",
                f"        intervals: size = {len(reconciled_intervals)}",
            ]
        )
        for idx, (xmin, xmax, label) in enumerate(reconciled_intervals, 1):
            lines.extend(
                [
                    f"        intervals [{idx}]:",
                    f"            xmin = {xmin:.3f}",
                    f"            xmax = {xmax:.3f}",
                    f'            text = "{label}"',
                ]
            )
        curr_item += 1

    lines.extend(
        [
            f"    item [{curr_item}]:",
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
        word_objs = []
        for w in v.words:
            w_dict = {
                "word": w.word,
                "start": w.start_sec,
                "end": w.end_sec,
                "confidence": w.confidence,
                "flagged": w.flagged,
            }
            if w.cherokee_syllabary:
                w_dict["syllabary_word"] = w.cherokee_syllabary
            word_objs.append(w_dict)

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
