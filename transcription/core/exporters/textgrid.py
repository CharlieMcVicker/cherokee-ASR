# -*- coding: utf-8 -*-
"""
textgrid.py

Language-agnostic multi-tier Praat TextGrid serialization and builder models.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from transcription.core.alignment.models import AlignmentOutput, WordInterval


@dataclass
class IntervalTier:
    """Represents a single IntervalTier in a Praat TextGrid."""

    name: str
    intervals: List[Tuple[float, float, str]] = field(default_factory=list)
    xmin: float = 0.0
    xmax: Optional[float] = None

    def add_interval(self, xmin: float, xmax: float, text: str) -> None:
        """Add an interval to the tier."""
        self.intervals.append((xmin, xmax, text))
        if self.xmax is None or xmax > self.xmax:
            self.xmax = xmax

    def format_lines(self, tier_idx: int, total_max_t: float) -> List[str]:
        """Formats the IntervalTier block according to Praat ooTextFile specification."""
        tier_xmax = total_max_t if self.xmax is None else max(self.xmax, total_max_t)
        t_lines = [
            f"    item [{tier_idx}]:",
            '        class = "IntervalTier"',
            f'        name = "{self.name}"',
            f"        xmin = {self.xmin:g}",
            f"        xmax = {tier_xmax:.3f}",
            f"        intervals: size = {len(self.intervals)}",
        ]
        for idx, (xmin, xmax, label) in enumerate(self.intervals, 1):
            t_lines.extend(
                [
                    f"        intervals [{idx}]:",
                    f"            xmin = {xmin:.3f}",
                    f"            xmax = {xmax:.3f}",
                    f'            text = "{label}"',
                ]
            )
        return t_lines


@dataclass
class TextGridBuilder:
    """Builder for constructing language-agnostic multi-tier Praat TextGrids."""

    xmin: float = 0.0
    xmax: float = 0.0
    tiers: List[IntervalTier] = field(default_factory=list)

    def add_tier(self, tier: IntervalTier) -> "TextGridBuilder":
        """Add an IntervalTier to the builder."""
        self.tiers.append(tier)
        tier_xmax = tier.xmax if tier.xmax is not None else 0.0
        if tier_xmax > self.xmax:
            self.xmax = tier_xmax
        return self

    def create_tier(
        self,
        name: str,
        intervals: Optional[List[Tuple[float, float, str]]] = None,
        xmin: float = 0.0,
        xmax: Optional[float] = None,
    ) -> IntervalTier:
        """Constructs, registers, and returns a new IntervalTier."""
        tier = IntervalTier(
            name=name,
            intervals=intervals if intervals is not None else [],
            xmin=xmin,
            xmax=xmax,
        )
        self.add_tier(tier)
        return tier

    def to_textgrid_string(self) -> str:
        """Serializes the builder content into standard Praat ooTextFile string format."""
        total_end = self.xmax if self.xmax > 0.0 else 1.0
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            f"xmin = {self.xmin:g}",
            f"xmax = {total_end:.3f}",
            "tiers? <exists>",
            f"size = {len(self.tiers)}",
            "item []:",
        ]
        for idx, tier in enumerate(self.tiers, 1):
            lines.extend(tier.format_lines(idx, total_end))
        return "\n".join(lines) + "\n"

    def write(self, output_path: Union[str, Path]) -> str:
        """Serializes and writes TextGrid file to output path."""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_textgrid_string()
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path_obj)


def build_contiguous_intervals(
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


def build_padded_word_intervals(
    word_raw: List[Dict[str, Any]], total_end: float, pad_sec: float = 0.10
) -> List[Tuple[float, float, str]]:
    """Builds padded word intervals with temporal fusion of overlapping boundaries."""
    valid_words = [w for w in word_raw if w["end_sec"] > w["start_sec"]]
    valid_words.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

    if not valid_words:
        return build_contiguous_intervals([], total_end)

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

    return build_contiguous_intervals(fused, total_end)


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
    output_path = Path(output_dir) / filename

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
    chunk_intervals = build_contiguous_intervals(chunk_raw, total_end)

    # Tier 2: Words
    word_raw = []
    for c in alignment.aligned_chunks:
        for w in c.words:
            word_raw.append(
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            )

    word_intervals = build_contiguous_intervals(word_raw, total_end)
    padded_word_intervals = build_padded_word_intervals(
        word_raw, total_end, pad_sec=pad_sec
    )

    builder = TextGridBuilder(xmin=0.0, xmax=total_end)
    builder.create_tier("Chunks", chunk_intervals)
    builder.create_tier("Words", word_intervals)
    builder.create_tier("Padded Words", padded_word_intervals)

    # Build additional custom tiers
    if additional_word_tiers:
        for tier_name, word_seq in additional_word_tiers.items():
            tier_raw = [
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
                for w in word_seq
                if w.word and w.word.strip()
            ]
            tier_intervals = build_contiguous_intervals(tier_raw, total_end)
            builder.create_tier(tier_name, tier_intervals)

    raw_token_items = [
        {
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "text": t.word,
        }
        for t in (alignment.raw_tokens or [])
    ]
    if raw_token_items:
        raw_token_intervals = build_contiguous_intervals(raw_token_items, total_end)
        builder.create_tier("Raw ASR Emissions", raw_token_intervals)

    return builder.write(output_path)
