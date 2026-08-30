"""
Outbound adapters exporting domain AlignmentOutput to Praat .TextGrid and alignment_manifest.json formats.
"""

import json
import os
from typing import Any, Dict, List, Tuple
from transcription.alignment.domain.models import AlignmentOutput


class PraatTextGridAdapter:
    """
    Adapter for exporting AlignmentOutput to Praat .TextGrid format.
    Generates contiguous IntervalTiers for:
    - Tier 1: Chunks / Verses
    - Tier 2: Words
    - Tier 3: Padded Words (+ pad_sec fusion)
    - Optional Tier: Reconciled Words (if present)
    - Final Tier: Raw ASR Emissions (if present)
    """

    def __init__(self, pad_sec: float = 0.10):
        self.pad_sec = pad_sec

    def _build_contiguous_intervals(
        self, raw_intervals: List[Dict[str, Any]], total_end: float
    ) -> List[Tuple[float, float, str]]:
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
        self, word_raw: List[Dict[str, Any]], total_end: float
    ) -> List[Tuple[float, float, str]]:
        valid_words = [w for w in word_raw if w["end_sec"] > w["start_sec"]]
        valid_words.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

        if not valid_words:
            return self._build_contiguous_intervals([], total_end)

        fused = []
        curr_start = valid_words[0]["start_sec"]
        curr_end = valid_words[0]["end_sec"] + self.pad_sec
        curr_texts = [valid_words[0]["text"]]

        for w in valid_words[1:]:
            next_start = w["start_sec"]
            next_end = w["end_sec"] + self.pad_sec
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

        return self._build_contiguous_intervals(fused, total_end)

    def export(self, alignment: AlignmentOutput, output_path: str) -> None:
        """
        Writes Praat .TextGrid file from AlignmentOutput.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        total_end = 0.0
        if alignment.aligned_chunks:
            total_end = max(c.end_sec for c in alignment.aligned_chunks)
        if alignment.raw_tokens:
            total_end = max(total_end, max(t.end_sec for t in alignment.raw_tokens))
        if total_end <= 0.0:
            total_end = 1.0

        # Tier 1: Chunks / Verses
        chunk_raw = [
            {
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "text": f"{c.chunk_id}: {c.chunk.raw_text}",
            }
            for c in alignment.aligned_chunks
        ]
        chunk_intervals = self._build_contiguous_intervals(chunk_raw, total_end)

        # Tier 2: Words
        word_raw = []
        for c in alignment.aligned_chunks:
            for w in c.words:
                word_raw.append(
                    {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
                )
        word_intervals = self._build_contiguous_intervals(word_raw, total_end)

        # Tier 3: Padded Words
        padded_word_intervals = self._build_padded_word_intervals(word_raw, total_end)

        # Check for reconciled tier
        reconciled_raw = []
        has_reconciled = False
        for c in alignment.aligned_chunks:
            for w in c.words:
                rec_text = w.reconciled_word or w.word
                if w.reconciled_word:
                    has_reconciled = True
                reconciled_raw.append(
                    {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": rec_text}
                )

        raw_token_items = [
            {
                "start_sec": t.start_sec,
                "end_sec": t.end_sec,
                "text": t.word,
            }
            for t in (alignment.raw_tokens or [])
        ]
        has_raw_tokens = bool(raw_token_items)

        tier_count = 3
        if has_reconciled:
            tier_count += 1
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
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "Chunks"',
            "        xmin = 0",
            f"        xmax = {total_end:.3f}",
            f"        intervals: size = {len(chunk_intervals)}",
        ]

        for idx, (xmin, xmax, label) in enumerate(chunk_intervals, 1):
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

        curr_tier_num = 4
        if has_reconciled:
            reconciled_intervals = self._build_contiguous_intervals(
                reconciled_raw, total_end
            )
            lines.extend(
                [
                    f"    item [{curr_tier_num}]:",
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
            curr_tier_num += 1

        if has_raw_tokens:
            raw_token_intervals = self._build_contiguous_intervals(
                raw_token_items, total_end
            )
            lines.extend(
                [
                    f"    item [{curr_tier_num}]:",
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


class ManifestJsonAdapter:
    """
    Adapter for exporting AlignmentOutput to alignment_manifest.json format.
    """

    def export(self, alignment: AlignmentOutput, output_path: str) -> None:
        """
        Writes alignment manifest JSON matching project schema.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        manifest_lines = []
        for c in alignment.aligned_chunks:
            word_objs = []
            for w in c.words:
                w_dict: Dict[str, Any] = {
                    "word": w.word,
                    "start": w.start_sec,
                    "end": w.end_sec,
                    "confidence": w.confidence,
                    "flagged": w.flagged,
                }
                if w.syllabary:
                    w_dict["syllabary_word"] = w.syllabary
                if w.reconciled_word:
                    w_dict["reconciled_word"] = w.reconciled_word
                word_objs.append(w_dict)

            meta = c.chunk.metadata
            line_dict: Dict[str, Any] = {
                "line_id": c.chunk_id,
                "cherokee_syllabary": c.chunk.syllabary_text or "",
                "text": c.chunk.raw_text,
                "english": meta.get("english", ""),
                "start": c.start_sec,
                "end": c.end_sec,
                "cer": c.distance_score,
                "emitted_text": c.emitted_text,
                "words": word_objs,
            }
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

        data = {
            "audio_source": alignment.source_id,
            "metrics": metrics_dict,
            "lines": manifest_lines,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
