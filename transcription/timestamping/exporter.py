# -*- coding: utf-8 -*-
"""
exporter.py

Praat TextGrid and JSON alignment_manifest.json export module.
"""

import json
import os
from typing import Dict, Any
from transcription.timestamping.aligner import AlignmentResult


def export_praat_textgrid(alignment: AlignmentResult, output_path: str) -> None:
    """
    Generates a valid dual-tier Praat .TextGrid file:
    - Tier 1: Verses (IntervalTier)
    - Tier 2: Words (IntervalTier)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine total audio end time
    total_end = 0.0
    if alignment.verses:
        total_end = max(v.end_sec for v in alignment.verses)
    if total_end <= 0.0:
        total_end = 1.0

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {total_end:.3f}",
        "tiers? <exists>",
        "size = 2",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "Verses"',
        "        xmin = 0",
        f"        xmax = {total_end:.3f}",
        f"        intervals: size = {len(alignment.verses)}",
    ]

    for idx, v in enumerate(alignment.verses, 1):
        text_label = f"{v.line_id}: {v.raw_phonetic}"
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {v.start_sec:.3f}",
                f"            xmax = {v.end_sec:.3f}",
                f'            text = "{text_label}"',
            ]
        )

    # Word Intervals
    all_words = []
    for v in alignment.verses:
        all_words.extend(v.words)

    lines.extend(
        [
            "    item [2]:",
            '        class = "IntervalTier"',
            '        name = "Words"',
            "        xmin = 0",
            f"        xmax = {total_end:.3f}",
            f"        intervals: size = {len(all_words)}",
        ]
    )

    for idx, w in enumerate(all_words, 1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {w.start_sec:.3f}",
                f"            xmax = {w.end_sec:.3f}",
                f'            text = "{w.word}"',
            ]
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_alignment_manifest(alignment: AlignmentResult, output_path: str) -> None:
    """
    Exports alignment_manifest.json following ground-truth app schema.
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
                "words": word_objs,
            }
        )

    data = {
        "audio_source": alignment.audio_source,
        "lines": manifest_lines,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
