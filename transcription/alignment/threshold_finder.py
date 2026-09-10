"""
Interactive CLI binary search tool for alignment cost thresholding.

This module provides tools to determine an optimal alignment cost threshold T*
via interactive binary search over the cost distribution of alignment records.
At each candidate threshold midpoint T_mid, sample verses are displayed to the
user for quality grading (Accept/Reject), rapidly converging on the decision boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Mapping of standard 2-digit New Testament book IDs to English book names
NT_BOOK_NAMES: Dict[int, str] = {
    1: "Matthew",
    2: "Mark",
    3: "Luke",
    4: "John",
    5: "Acts",
    6: "Romans",
    7: "1 Corinthians",
    8: "2 Corinthians",
    9: "Galatians",
    10: "Ephesians",
    11: "Philippians",
    12: "Colossians",
    13: "1 Thessalonians",
    14: "2 Thessalonians",
    15: "1 Timothy",
    16: "2 Timothy",
    17: "Titus",
    18: "Philemon",
    19: "Hebrews",
    20: "James",
    21: "1 Peter",
    22: "2 Peter",
    23: "1 John",
    24: "2 John",
    25: "3 John",
    26: "Jude",
    27: "Revelation",
}


def parse_verse_reference(verse_id: str) -> str:
    """
    Format a raw verse/line ID into a human-readable verse reference.

    Examples:
        '020114' -> 'Mark 01:14 (020114)'
        '010503' -> 'Matthew 05:03 (010503)'
        'mark_01_14' -> 'Mark 01:14 (mark_01_14)'
        'chunk_042' -> 'chunk_042'
    """
    cleaned = str(verse_id).strip()
    if not cleaned:
        return "Unknown"

    # 6-digit standard Bible ID: BB CC VV (Book, Chapter, Verse)
    if cleaned.isdigit() and len(cleaned) == 6:
        book_num = int(cleaned[0:2])
        chapter_num = int(cleaned[2:4])
        verse_num = int(cleaned[4:6])
        book_name = NT_BOOK_NAMES.get(book_num, f"Book {book_num}")
        return f"{book_name} {chapter_num:02d}:{verse_num:02d} ({cleaned})"

    # book_chapter_verse pattern (e.g. mark_01_14)
    parts = cleaned.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        book_str = " ".join(parts[:-2]).title()
        return f"{book_str} {int(parts[-2]):02d}:{int(parts[-1]):02d} ({cleaned})"

    return cleaned


@dataclass
class AlignmentRecord:
    """Standardized alignment record representation for threshold exploration."""

    verse_id: str
    cherokee_syllabary: str
    text: str
    emitted_text: str
    cost: float
    audio_path: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    english: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_sec <= 0.0 and self.end_sec > self.start_sec:
            self.duration_sec = round(self.end_sec - self.start_sec, 3)

    @property
    def formatted_verse_id(self) -> str:
        """Formatted human-readable reference for the verse ID."""
        return parse_verse_reference(self.verse_id)

    @property
    def audio_filename(self) -> str:
        """Audio source base filename."""
        return os.path.basename(self.audio_path) if self.audio_path else "N/A"


@dataclass
class ThresholdSearchStep:
    """Metadata recorded for an individual binary search iteration step."""

    iteration: int
    t_low: float
    t_high: float
    t_mid: float
    action: str
    sampled_verse_ids: List[str] = field(default_factory=list)


@dataclass
class ThresholdMetrics:
    """Summary metrics and statistical evaluation for a selected threshold T*."""

    threshold: float
    total_verses: int
    accepted_count: int
    rejected_count: int
    acceptance_rate: float
    percentile: float
    cost_min: float
    cost_max: float
    cost_mean: float
    cost_median: float
    mean_accepted_cost: float
    mean_rejected_cost: float
    t_low: float
    t_high: float
    tolerance: float
    iterations: int
    timestamp: str
    source_files: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary structure for JSON serialization."""
        return asdict(self)


def _extract_record_from_dict(
    d: Dict[str, Any], default_audio: str = ""
) -> Optional[AlignmentRecord]:
    """Helper to parse an alignment record dictionary with flexible key naming."""
    # Find ID
    verse_id = (
        d.get("line_id")
        or d.get("verse_id")
        or d.get("id")
        or d.get("chunk_id")
        or d.get("name")
        or ""
    )
    verse_id = str(verse_id).strip()

    # Find Cost / CER / Distance score
    raw_cost = (
        d.get("cer")
        if "cer" in d
        else (
            d.get("cost")
            if "cost" in d
            else (
                d.get("distance_score")
                if "distance_score" in d
                else d.get("score") if "score" in d else d.get("error_rate")
            )
        )
    )
    if raw_cost is None:
        return None
    try:
        cost = float(raw_cost)
    except (ValueError, TypeError):
        return None

    # Text fields
    cherokee = (
        d.get("cherokee_syllabary")
        or d.get("reference_sentence")
        or d.get("reference_syllabary")
        or d.get("cherokee")
        or d.get("syllabary")
        or d.get("syllabary_text")
        or ""
    )
    text = (
        d.get("reconciled_phonetics")
        or d.get("reference_phonetic")
        or d.get("text")
        or d.get("reconciled_text")
        or d.get("phonetics")
        or d.get("reference_text")
        or d.get("ground_truth")
        or ""
    )
    emitted = (
        d.get("asr_hypothesis")
        or d.get("emitted_text")
        or d.get("asr_text")
        or d.get("hypothesis")
        or d.get("prediction")
        or d.get("transcript")
        or ""
    )
    english = d.get("english") or ""

    # Audio & Timestamps
    audio_path = (
        d.get("audio_path")
        or d.get("audio_source")
        or d.get("filepath")
        or d.get("path")
        or d.get("wav_path")
        or default_audio
    )
    try:
        start_sec = float(
            d.get("start") or d.get("start_sec") or d.get("start_seconds") or 0.0
        )
    except (ValueError, TypeError):
        start_sec = 0.0

    try:
        end_sec = float(d.get("end") or d.get("end_sec") or d.get("end_seconds") or 0.0)
    except (ValueError, TypeError):
        end_sec = 0.0

    try:
        duration_sec = float(
            d.get("duration")
            or d.get("duration_sec")
            or d.get("duration_seconds")
            or 0.0
        )
    except (ValueError, TypeError):
        duration_sec = 0.0

    return AlignmentRecord(
        verse_id=str(verse_id),
        cherokee_syllabary=str(cherokee).strip(),
        text=str(text).strip(),
        emitted_text=str(emitted).strip(),
        cost=cost,
        audio_path=str(audio_path).strip(),
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=duration_sec,
        english=str(english).strip(),
        metadata={
            k: v for k, v in d.items() if k not in ("words", "additional_word_tiers")
        },
    )


def load_alignment_records(
    source: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[List[AlignmentRecord], List[str]]:
    """
    Load alignment records from a file path, directory of manifests, or in-memory structures.

    Returns:
        Tuple of (records_list, loaded_filepaths_list)
    """
    records: List[AlignmentRecord] = []
    loaded_files: List[str] = []

    if isinstance(source, (list, tuple)):
        for item in source:
            if isinstance(item, dict):
                # Check if it's a manifest dict
                if "lines" in item and isinstance(item["lines"], list):
                    audio_src = str(item.get("audio_source", ""))
                    for line_dict in item["lines"]:
                        rec = _extract_record_from_dict(
                            line_dict, default_audio=audio_src
                        )
                        if rec:
                            records.append(rec)
                else:
                    rec = _extract_record_from_dict(item)
                    if rec:
                        records.append(rec)
        return records, loaded_files

    if isinstance(source, dict):
        if "lines" in source and isinstance(source["lines"], list):
            audio_src = str(source.get("audio_source", ""))
            for line_dict in source["lines"]:
                rec = _extract_record_from_dict(line_dict, default_audio=audio_src)
                if rec:
                    records.append(rec)
        elif "records" in source and isinstance(source["records"], list):
            for line_dict in source["records"]:
                rec = _extract_record_from_dict(line_dict)
                if rec:
                    records.append(rec)
        else:
            # Map of verse_id -> dict
            for key, val in source.items():
                if isinstance(val, dict):
                    d = dict(val)
                    if "line_id" not in d and "verse_id" not in d:
                        d["line_id"] = key
                    rec = _extract_record_from_dict(d)
                    if rec:
                        records.append(rec)
        return records, loaded_files

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Alignment source path does not exist: {path}")

    if path.is_dir():
        # Scan directory for alignment manifest JSON files
        manifest_files = sorted(path.rglob("*manifest*.json"))
        if not manifest_files:
            manifest_files = sorted(path.rglob("*.json"))

        for mf in manifest_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_recs, _ = load_alignment_records(data)
                records.extend(file_recs)
                loaded_files.append(str(mf))
            except Exception as err:
                print(f"Warning: Could not parse {mf}: {err}", file=sys.stderr)

        return records, loaded_files

    # Single file
    loaded_files.append(str(path))
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_recs, _ = load_alignment_records(data)
        records.extend(file_recs)
    elif path.suffix.lower() in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                rec = _extract_record_from_dict(row)
                if rec:
                    records.append(rec)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return records, loaded_files


def find_default_dataset_path() -> Optional[str]:
    """Auto-discover standard alignment manifests or dataset records in the workspace."""
    candidates = [
        "cherokee_new_testament/alignments/bible_alignment_records.json",
        "output_praat/new_testament",
        "data/results/alignment_mark_01/alignment_manifest.json",
        "output_praat/new_testament/mark_01/alignment_manifest.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class AlignmentThresholdFinder:
    """
    Binary search optimizer over alignment cost distributions.

    Maintains bounds [T_low, T_high] and samples candidate verses at T_mid = (T_low + T_high) / 2.
    Supports interactive terminal prompt or programmatic headless callbacks.
    """

    def __init__(
        self,
        records: List[AlignmentRecord],
        k_samples: int = 3,
        tolerance: float = 0.005,
        max_iter: int = 15,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        source_files: Optional[List[str]] = None,
    ) -> None:
        if not records:
            raise ValueError(
                "Cannot initialize AlignmentThresholdFinder with empty records list."
            )

        self.records = records
        self.k_samples = max(1, k_samples)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.source_files = list(source_files or [])

        # Extract sorted costs
        self.costs = sorted([r.cost for r in records])
        self.dataset_min = self.costs[0]
        self.dataset_max = self.costs[-1]

        # Initialize bounds
        self.t_low = self.dataset_min if min_cost is None else float(min_cost)
        self.t_high = self.dataset_max if max_cost is None else float(max_cost)
        if self.t_low > self.t_high:
            self.t_low, self.t_high = self.t_high, self.t_low

        self.history: List[ThresholdSearchStep] = []
        self._displayed_verse_ids: Set[str] = set()

    def sample_near_cost(
        self,
        target_cost: float,
        k: Optional[int] = None,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[AlignmentRecord]:
        """
        Sample K verses whose alignment cost is closest to target_cost.
        """
        sample_count = k if k is not None else self.k_samples
        exclude = exclude_ids if exclude_ids is not None else set()

        available = [r for r in self.records if r.verse_id not in exclude]
        if not available:
            # Fall back to all records if everything was excluded
            available = self.records

        # Sort by distance to target cost
        sorted_by_dist = sorted(
            available, key=lambda r: (abs(r.cost - target_cost), r.cost)
        )
        return sorted_by_dist[:sample_count]

    def format_record_display(
        self, record: AlignmentRecord, index: Optional[int] = None
    ) -> str:
        """Format a single alignment record for terminal display."""
        prefix = f" [{index}]" if index is not None else " [•]"
        time_str = ""
        if record.end_sec > record.start_sec:
            time_str = f" ({record.start_sec:.2f}s - {record.end_sec:.2f}s, dur: {record.duration_sec:.2f}s)"

        lines = [
            f"{prefix} Verse: {record.formatted_verse_id} | Cost: {record.cost:.4f} | Audio: {record.audio_filename}{time_str}",
        ]
        if record.cherokee_syllabary:
            lines.append(f"     Syllabary  : {record.cherokee_syllabary}")
        if record.text:
            lines.append(f"     Reconciled : {record.text}")
        if record.emitted_text:
            lines.append(f"     Hypothesis : {record.emitted_text}")
        if record.english:
            lines.append(f"     English    : {record.english}")

        return "\n".join(lines)

    def format_iteration_prompt(
        self,
        t_low: float,
        t_high: float,
        t_mid: float,
        samples: List[AlignmentRecord],
        iteration: int,
    ) -> str:
        """Construct the prompt header, sampled verses, and command options."""
        percentile = self.calculate_percentile(t_mid)
        sep = "=" * 70
        sub_sep = "-" * 70

        blocks = [
            sep,
            f" Iteration {iteration}/{self.max_iter} | Bounds: [T_low={t_low:.4f}, T_high={t_high:.4f}] (Diff: {t_high - t_low:.4f})",
            f" Candidate Threshold T_mid: {t_mid:.4f} (~{percentile:.1f}% percentile in dataset)",
            f" Sampled {len(samples)} Candidate Verses Closest to T_mid:",
            sep,
        ]

        for idx, rec in enumerate(samples, start=1):
            blocks.append(self.format_record_display(rec, index=idx))
            if idx < len(samples):
                blocks.append(sub_sep)

        blocks.extend(
            [
                sep,
                " Commands:",
                f"  [A]ccept / Good  : Quality acceptable -> increase lower bound (T_low = {t_mid:.4f})",
                f"  [R]eject / Bad   : Quality unacceptable / noisy -> decrease upper bound (T_high = {t_mid:.4f})",
                "  [U]nsure / Skip  : Sample alternate verses near this cost level",
                "  [S]et <value>    : Manually set explicit threshold (e.g. 's 0.15')",
                f"  [D]one / Confirm : Finalize current threshold T* = {t_mid:.4f}",
                "  [Q]uit           : Exit search",
                sep,
            ]
        )
        return "\n".join(blocks)

    def calculate_percentile(self, threshold: float) -> float:
        """Calculate the percentage of records with cost <= threshold."""
        if not self.costs:
            return 0.0
        count = sum(1 for c in self.costs if c <= threshold)
        return (count / len(self.costs)) * 100.0

    def compute_metrics(self, threshold: float) -> ThresholdMetrics:
        """Compute comprehensive dataset statistics for a chosen threshold T*."""
        total = len(self.records)
        accepted = [r for r in self.records if r.cost <= threshold]
        rejected = [r for r in self.records if r.cost > threshold]

        acc_count = len(accepted)
        rej_count = len(rejected)
        acc_rate = (acc_count / total) if total > 0 else 0.0
        percentile = acc_rate * 100.0

        mean_cost = sum(self.costs) / total if total > 0 else 0.0
        median_cost = self.costs[total // 2] if total > 0 else 0.0

        mean_acc = (sum(r.cost for r in accepted) / acc_count) if acc_count > 0 else 0.0
        mean_rej = (sum(r.cost for r in rejected) / rej_count) if rej_count > 0 else 0.0

        return ThresholdMetrics(
            threshold=round(threshold, 6),
            total_verses=total,
            accepted_count=acc_count,
            rejected_count=rej_count,
            acceptance_rate=round(acc_rate, 4),
            percentile=round(percentile, 2),
            cost_min=round(self.dataset_min, 4),
            cost_max=round(self.dataset_max, 4),
            cost_mean=round(mean_cost, 4),
            cost_median=round(median_cost, 4),
            mean_accepted_cost=round(mean_acc, 4),
            mean_rejected_cost=round(mean_rej, 4),
            t_low=round(self.t_low, 6),
            t_high=round(self.t_high, 6),
            tolerance=self.tolerance,
            iterations=len(self.history),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_files=self.source_files,
            history=[asdict(step) for step in self.history],
        )

    def export_results(
        self,
        metrics: ThresholdMetrics,
        output_path: Union[str, Path] = "runs/evaluation/alignment_threshold.json",
    ) -> Path:
        """Export calculated threshold configuration and summary metrics to a JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        return out_p

    def run(
        self,
        prompt_callback: Optional[
            Callable[[Dict[str, Any], List[AlignmentRecord]], str]
        ] = None,
        threshold: Optional[float] = None,
        quantile: Optional[float] = None,
        quiet: bool = False,
    ) -> ThresholdMetrics:
        """
        Execute threshold search loop.

        Args:
            prompt_callback: Optional callable for automated/headless execution.
                             Receives state dict and sampled records, returns command string.
            threshold: Direct threshold override (headless).
            quantile: Direct quantile override (0.0 to 1.0) (headless).
            quiet: If True, suppress stdout printing during search.

        Returns:
            ThresholdMetrics containing results and metadata.
        """
        # Headless direct threshold
        if threshold is not None:
            t_val = float(threshold)
            self.t_low = t_val
            self.t_high = t_val
            metrics = self.compute_metrics(t_val)
            if not quiet:
                print(
                    f"Direct threshold specified: T* = {t_val:.4f} (Accepted: {metrics.accepted_count}/{metrics.total_verses}, {metrics.percentile:.1f}%)"
                )
            return metrics

        # Headless quantile
        if quantile is not None:
            q = max(0.0, min(1.0, float(quantile)))
            idx = min(len(self.costs) - 1, int(math.floor(q * (len(self.costs) - 1))))
            t_val = self.costs[idx]
            self.t_low = t_val
            self.t_high = t_val
            metrics = self.compute_metrics(t_val)
            if not quiet:
                print(
                    f"Quantile threshold specified ({q:.2%}): T* = {t_val:.4f} (Accepted: {metrics.accepted_count}/{metrics.total_verses}, {metrics.percentile:.1f}%)"
                )
            return metrics

        # Interactive / Callback Search Loop
        iteration = 0
        final_threshold: Optional[float] = None
        skip_sample_ids: Set[str] = set()

        while iteration < self.max_iter:
            iteration += 1
            t_mid = (self.t_low + self.t_high) / 2.0

            # Check convergence
            if (self.t_high - self.t_low) <= self.tolerance:
                if not quiet:
                    print(
                        f"\n[✓] Search converged within tolerance {self.tolerance:.4f} (Bounds: [{self.t_low:.4f}, {self.t_high:.4f}])."
                    )
                final_threshold = t_mid
                break

            # Sample candidate verses
            samples = self.sample_near_cost(
                target_cost=t_mid,
                k=self.k_samples,
                exclude_ids=skip_sample_ids,
            )
            sampled_ids = [s.verse_id for s in samples]
            self._displayed_verse_ids.update(sampled_ids)

            if not quiet:
                prompt_text = self.format_iteration_prompt(
                    t_low=self.t_low,
                    t_high=self.t_high,
                    t_mid=t_mid,
                    samples=samples,
                    iteration=iteration,
                )
                print(prompt_text)

            # Get user response (callback or interactive input)
            if prompt_callback is not None:
                state = {
                    "iteration": iteration,
                    "t_low": self.t_low,
                    "t_high": self.t_high,
                    "t_mid": t_mid,
                    "history": self.history,
                }
                raw_input = prompt_callback(state, samples)
            else:
                try:
                    raw_input = input(" Choice [A/R/U/S/D/Q]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nSearch interrupted.")
                    final_threshold = t_mid
                    break

            cmd = raw_input.strip().lower()

            # Process command
            if cmd in ("a", "accept", "good", "g", "y", "yes", "1"):
                self.history.append(
                    ThresholdSearchStep(
                        iteration=iteration,
                        t_low=self.t_low,
                        t_high=self.t_high,
                        t_mid=t_mid,
                        action="accept",
                        sampled_verse_ids=sampled_ids,
                    )
                )
                self.t_low = t_mid
                skip_sample_ids.clear()
            elif cmd in ("r", "reject", "bad", "b", "n", "no", "0"):
                self.history.append(
                    ThresholdSearchStep(
                        iteration=iteration,
                        t_low=self.t_low,
                        t_high=self.t_high,
                        t_mid=t_mid,
                        action="reject",
                        sampled_verse_ids=sampled_ids,
                    )
                )
                self.t_high = t_mid
                skip_sample_ids.clear()
            elif cmd in ("u", "unsure", "skip", "k"):
                self.history.append(
                    ThresholdSearchStep(
                        iteration=iteration,
                        t_low=self.t_low,
                        t_high=self.t_high,
                        t_mid=t_mid,
                        action="unsure",
                        sampled_verse_ids=sampled_ids,
                    )
                )
                skip_sample_ids.update(sampled_ids)
                if not quiet:
                    print(f"Skipping sampled verses; resampling near {t_mid:.4f}...")
            elif cmd.startswith("s ") or cmd.startswith("set "):
                val_str = cmd.split(maxsplit=1)[1].strip()
                try:
                    manual_val = float(val_str)
                    if manual_val < 0.0:
                        raise ValueError("Threshold cannot be negative.")
                    self.history.append(
                        ThresholdSearchStep(
                            iteration=iteration,
                            t_low=self.t_low,
                            t_high=self.t_high,
                            t_mid=manual_val,
                            action=f"set_{manual_val}",
                            sampled_verse_ids=sampled_ids,
                        )
                    )
                    self.t_low = manual_val
                    self.t_high = manual_val
                    final_threshold = manual_val
                    if not quiet:
                        print(f"Manually set threshold T* = {manual_val:.4f}")
                    break
                except ValueError as err:
                    if not quiet:
                        print(f"Invalid threshold value '{val_str}': {err}")
                    iteration -= 1  # Do not count invalid input as an iteration
            elif cmd in ("d", "done", "confirm", "c"):
                final_threshold = t_mid
                self.history.append(
                    ThresholdSearchStep(
                        iteration=iteration,
                        t_low=self.t_low,
                        t_high=self.t_high,
                        t_mid=t_mid,
                        action="done",
                        sampled_verse_ids=sampled_ids,
                    )
                )
                if not quiet:
                    print(f"Confirmed threshold T* = {t_mid:.4f}")
                break
            elif cmd in ("q", "quit", "exit"):
                if not quiet:
                    print("Exiting search without locking threshold.")
                final_threshold = t_mid
                break
            else:
                if not quiet:
                    print(
                        f"Unknown command '{raw_input}'. Please choose [A]ccept, [R]eject, [U]nsure, [S]et <val>, [D]one, or [Q]uit."
                    )
                iteration -= 1

        if final_threshold is None:
            final_threshold = (self.t_low + self.t_high) / 2.0

        metrics = self.compute_metrics(final_threshold)
        return metrics


def find_threshold_bounds(
    records: List[AlignmentRecord],
    target_acceptance_rate: float = 0.90,
) -> Tuple[float, ThresholdMetrics]:
    """
    Programmatically find the alignment cost threshold that achieves a target acceptance rate.
    """
    finder = AlignmentThresholdFinder(records)
    metrics = finder.run(quantile=target_acceptance_rate, quiet=True)
    return metrics.threshold, metrics


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for alignment cost threshold finder."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI Binary Search Tool for Alignment Cost Thresholding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to alignment manifest JSON, dataset records CSV/JSON, or directory of manifests. Auto-discovers default if omitted.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="runs/evaluation/alignment_threshold.json",
        help="Output path for threshold configuration JSON.",
    )
    parser.add_argument(
        "--samples",
        "-k",
        type=int,
        default=3,
        help="Number of candidate verses to display per iteration.",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=0.005,
        help="Convergence tolerance epsilon for bracket narrowing.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Maximum binary search iterations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Headless mode: directly specify threshold value T*.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="Headless mode: compute threshold by cost quantile (0.0 to 1.0).",
    )
    parser.add_argument(
        "--min-cost",
        type=float,
        default=None,
        help="Initial search lower bound T_low (default: minimum cost in dataset).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Initial search upper bound T_high (default: maximum cost in dataset).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress interactive outputs (useful for scripting).",
    )

    args = parser.parse_args(argv)

    input_path = args.input
    if input_path is None:
        input_path = find_default_dataset_path()
        if input_path is None:
            print(
                "Error: No input path specified and no default dataset found. Please provide --input <path>.",
                file=sys.stderr,
            )
            return 1
        print(f"[•] Auto-discovered dataset: {input_path}")

    try:
        records, source_files = load_alignment_records(input_path)
    except Exception as err:
        print(f"Error loading records from {input_path}: {err}", file=sys.stderr)
        return 1

    if not records:
        print(
            f"Error: No valid alignment records found in {input_path}.", file=sys.stderr
        )
        return 1

    print(
        f"[•] Loaded {len(records)} alignment records across {len(source_files)} source files."
    )
    print(
        f"[•] Cost Range: min={min(r.cost for r in records):.4f}, max={max(r.cost for r in records):.4f}, mean={sum(r.cost for r in records)/len(records):.4f}"
    )

    finder = AlignmentThresholdFinder(
        records=records,
        k_samples=args.samples,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
        min_cost=args.min_cost,
        max_cost=args.max_cost,
        source_files=source_files,
    )

    metrics = finder.run(
        threshold=args.threshold,
        quantile=args.quantile,
        quiet=args.quiet,
    )

    out_file = finder.export_results(metrics, output_path=args.output)
    print(f"\n[✓] Alignment threshold configuration saved to: {out_file}")
    print(f"    Selected Threshold T* : {metrics.threshold:.4f}")
    print(
        f"    Accepted Verses       : {metrics.accepted_count} / {metrics.total_verses} ({metrics.percentile:.1f}%)"
    )
    print(
        f"    Rejected Verses       : {metrics.rejected_count} / {metrics.total_verses}"
    )
    print(f"    Mean Accepted Cost    : {metrics.mean_accepted_cost:.4f}")
    print(f"    Mean Rejected Cost    : {metrics.mean_rejected_cost:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
