#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analyze_comparative_errors.py

Comprehensive 3-Way Comparative Error Analysis:
Evaluates Ground Truth vs. Greedy ASR vs. Syllabary-Guided CTC Segmentation
across the Cherokee Syllabary dataset (split_audio_syl_target.csv, 1,389 samples).
Categorizes all residual errors into granular phonological and orthographic taxonomies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import difflib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jiwer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_comparative_errors")


@dataclass(frozen=True)
class ErrorEvent:
    category: str
    gt_chunk: str
    hyp_chunk: str
    description: str
    sample_id: str
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreeWayComparisonResult:
    sample_id: str
    split: str
    syllabary: str
    template: str
    ground_truth: str
    greedy_hyp: str
    greedy_conf: float
    greedy_cer: float
    guided_hyp: str
    guided_conf: float
    guided_cer: float
    guided_events: List[ErrorEvent]
    greedy_events: List[ErrorEvent]
    winner: str  # "guided", "greedy", "tie"
    exact_status: str  # "both_exact", "guided_only", "greedy_only", "neither"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["guided_events"] = [e.to_dict() for e in self.guided_events]
        d["greedy_events"] = [e.to_dict() for e in self.greedy_events]
        return d


def classify_mismatch_sequence(
    gt: str,
    hyp: str,
    template: str,
    sample_id: str,
    split: str,
) -> List[ErrorEvent]:
    """
    Classifies discrepancies between ground truth (reference) and hypothesis string
    into discrete phonological, orthographic, and transcript categories.
    """
    if gt == hyp:
        return []

    events: List[ErrorEvent] = []
    matcher = difflib.SequenceMatcher(None, gt, hyp)

    gt_words = gt.split()
    hyp_words = hyp.split()
    temp_words = template.split()

    # Major word count or multi-word transcript discordance
    if len(gt_words) != len(temp_words):
        events.append(
            ErrorEvent(
                category="TRANSCRIPT_GT_DISCORDANCE",
                gt_chunk=gt,
                hyp_chunk=hyp,
                description=f"Word count discordance ({len(gt_words)} GT vs {len(temp_words)} template)",
                sample_id=sample_id,
                split=split,
            )
        )
        return events

    vowels = "aeiouv"
    sonorants = "nwyl"

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        gt_c = gt[i1:i2]
        hyp_c = hyp[j1:j2]

        prev_gt_char = gt[i1 - 1] if i1 > 0 else " "
        next_gt_char = gt[i2] if i2 < len(gt) else " "
        prev_hyp_char = hyp[j1 - 1] if j1 > 0 else " "
        next_hyp_char = hyp[j2] if j2 < len(hyp) else " "

        is_word_initial = prev_gt_char == " " or prev_hyp_char == " "
        is_word_final = next_gt_char == " " or next_hyp_char == " "

        # 1. Initial Sibilant Orthography (hs- vs s-)
        if is_word_initial and (
            (gt_c == "s" and hyp_c == "hs")
            or (gt_c == "hs" and hyp_c == "s")
            or (gt_c == "" and hyp_c == "h" and hyp[j2 : j2 + 1] == "s")
            or (gt_c == "h" and hyp_c == "" and gt[i2 : i2 + 1] == "s")
        ):
            events.append(
                ErrorEvent(
                    category="INITIAL_SIBILANT_HS_VS_S",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description="Word-initial sibilant orthography (s- vs hs-)",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 2. Hiatus / Glottal Stop Marker (')
        if "'" in gt_c or "'" in hyp_c:
            events.append(
                ErrorEvent(
                    category="GLOTTAL_STOP_HIATUS",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Glottal stop / hiatus apostrophe ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 3. Laryngeal Stride / Syllable Vowel Intrusion around h (akhahth vs akhth, kalhih vs kalh, uwhvh vs uwh)
        if ("h" in gt_c and any(v in gt_c for v in vowels) and hyp_c == "") or (
            "h" in hyp_c and any(v in hyp_c for v in vowels) and gt_c == ""
        ):
            events.append(
                ErrorEvent(
                    category="LARYNGEAL_STRIDE_VOWEL_INTRUSION",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Laryngeal stride vowel intrusion/syncope ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 4. Sonorant Pre-Aspiration & Devoicing (nh, wh, yh, lh)
        if (
            (gt_c == "h" and (prev_gt_char in sonorants or next_gt_char in sonorants))
            or (
                hyp_c == "h"
                and (prev_hyp_char in sonorants or next_hyp_char in sonorants)
            )
            or any(s in gt_c or s in hyp_c for s in ["nh", "wh", "yh", "lh"])
        ):
            events.append(
                ErrorEvent(
                    category="SONORANT_PREASPIRATION",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Sonorant pre-aspiration / devoicing ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 5. Stop & Affricate Aspiration (th/t, kh/k, tsh/ts, tlh/tl)
        if (
            (gt_c == "h" and (prev_gt_char in "kt" or next_gt_char in "kt"))
            or (hyp_c == "h" and (prev_hyp_char in "kt" or next_hyp_char in "kt"))
            or (gt_c in ["th", "kh", "tsh", "tlh"] and hyp_c in ["t", "k", "ts", "tl"])
            or (gt_c in ["t", "k", "ts", "tl"] and hyp_c in ["th", "kh", "tsh", "tlh"])
        ):
            events.append(
                ErrorEvent(
                    category="STOP_AFFRICATE_ASPIRATION",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Stop/affricate aspiration mismatch ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 6. Medial Sibilant Pre-Aspiration (hs vs s in medial/coda)
        if (
            (gt_c == "h" and (prev_gt_char == "s" or next_gt_char == "s"))
            or (hyp_c == "h" and (prev_hyp_char == "s" or next_hyp_char == "s"))
            or (gt_c == "hs" and hyp_c == "s")
            or (gt_c == "s" and hyp_c == "hs")
        ):
            events.append(
                ErrorEvent(
                    category="MEDIAL_SIBILANT_PREASPIRATION",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Medial sibilant pre-aspiration ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 7. Lateral Cluster Alternation (tl, tlh, lh, dl, l)
        if any(lat in gt_c or lat in hyp_c for lat in ["tl", "tlh", "lh", "dl", "l"]):
            events.append(
                ErrorEvent(
                    category="LATERAL_ALTERNATION",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Lateral cluster variation ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 8. Vowel Syncope Under-syncope (Guided retained full vowel from template, speaker dropped)
        if gt_c == "" and all(c in vowels for c in hyp_c):
            pos = (
                "FINAL"
                if is_word_final
                else ("INITIAL" if is_word_initial else "MEDIAL")
            )
            events.append(
                ErrorEvent(
                    category=f"VOWEL_UNDER_SYNCOPE_{pos}",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Under-syncope ({pos.lower()} vowel retained from template: {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 9. Vowel Syncope Over-syncope (Guided dropped vowel, speaker pronounced)
        if all(c in vowels for c in gt_c) and hyp_c == "":
            pos = (
                "FINAL"
                if is_word_final
                else ("INITIAL" if is_word_initial else "MEDIAL")
            )
            events.append(
                ErrorEvent(
                    category=f"VOWEL_OVER_SYNCOPE_{pos}",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Over-syncope ({pos.lower()} vowel deleted: {gt_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 10. Stray / Intervocalic Isolated Laryngeal H
        if (gt_c == "h" and hyp_c == "") or (gt_c == "" and hyp_c == "h"):
            events.append(
                ErrorEvent(
                    category="ISOLATED_LARYNGEAL_H",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Isolated laryngeal h ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 11. Multi-character Transcript Discordance
        if len(gt_c) >= 3 or len(hyp_c) >= 3:
            events.append(
                ErrorEvent(
                    category="TRANSCRIPT_GT_DISCORDANCE",
                    gt_chunk=gt_c,
                    hyp_chunk=hyp_c,
                    description=f"Multi-character divergence ({gt_c!r} -> {hyp_c!r})",
                    sample_id=sample_id,
                    split=split,
                )
            )
            continue

        # 12. Residual Substitution
        events.append(
            ErrorEvent(
                category="RESIDUAL_SUBSTITUTION",
                gt_chunk=gt_c,
                hyp_chunk=hyp_c,
                description=f"Residual single-char substitution ({gt_c!r} -> {hyp_c!r})",
                sample_id=sample_id,
                split=split,
            )
        )

    return events


def analyze_dataset_condition(
    samples_data: Sequence[Dict[str, Any]],
    condition_name: str,
) -> Tuple[List[ThreeWayComparisonResult], Dict[str, Any]]:
    """
    Performs full 3-way analysis for a single evaluation condition (Clean or Noisy).
    """
    results: List[ThreeWayComparisonResult] = []
    category_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    split_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "both_exact": 0,
            "guided_only_exact": 0,
            "greedy_only_exact": 0,
            "neither_exact": 0,
            "guided_better_cer": 0,
            "greedy_better_cer": 0,
            "equal_cer": 0,
        }
    )

    for s in samples_data:
        sample_id = s["sample_id"]
        split = s["split"]
        syllabary = s.get("syllabary", "")
        template = s["template"]
        gt = s["ground_truth"]
        greedy_hyp = s["greedy_hyp"]
        greedy_conf = s.get("greedy_conf", 0.0)
        guided_hyp = s["guided_hyp"]
        guided_conf = s.get("guided_conf", 0.0)

        gr_cer = round(jiwer.process_characters(gt, greedy_hyp).cer, 4)
        gd_cer = round(jiwer.process_characters(gt, guided_hyp).cer, 4)

        guided_events = classify_mismatch_sequence(
            gt, guided_hyp, template, sample_id, split
        )
        greedy_events = classify_mismatch_sequence(
            gt, greedy_hyp, template, sample_id, split
        )

        # Win / Loss / Exact Determination
        if guided_hyp == gt and greedy_hyp == gt:
            exact_status = "both_exact"
        elif guided_hyp == gt:
            exact_status = "guided_only"
        elif greedy_hyp == gt:
            exact_status = "greedy_only"
        else:
            exact_status = "neither"

        if gd_cer < gr_cer:
            winner = "guided"
        elif gr_cer < gd_cer:
            winner = "greedy"
        else:
            winner = "tie"

        res = ThreeWayComparisonResult(
            sample_id=sample_id,
            split=split,
            syllabary=syllabary,
            template=template,
            ground_truth=gt,
            greedy_hyp=greedy_hyp,
            greedy_conf=greedy_conf,
            greedy_cer=gr_cer,
            guided_hyp=guided_hyp,
            guided_conf=guided_conf,
            guided_cer=gd_cer,
            guided_events=guided_events,
            greedy_events=greedy_events,
            winner=winner,
            exact_status=exact_status,
        )
        results.append(res)

        # Aggregate category counts
        for ev in guided_events:
            category_counts["guided"][ev.category] += 1
            category_counts[f"guided_{split}"][ev.category] += 1
        for ev in greedy_events:
            category_counts["greedy"][ev.category] += 1
            category_counts[f"greedy_{split}"][ev.category] += 1

        # Aggregate 3-way split stats
        for s_key in [split, "all"]:
            st = split_stats[s_key]
            st["total"] += 1
            if exact_status == "both_exact":
                st["both_exact"] += 1
            elif exact_status == "guided_only":
                st["guided_only_exact"] += 1
            elif exact_status == "greedy_only":
                st["greedy_only_exact"] += 1
            else:
                st["neither_exact"] += 1

            if winner == "guided":
                st["guided_better_cer"] += 1
            elif winner == "greedy":
                st["greedy_better_cer"] += 1
            else:
                st["equal_cer"] += 1

    summary = {
        "condition": condition_name,
        "split_stats": dict(split_stats),
        "guided_error_categories": dict(category_counts["guided"]),
        "greedy_error_categories": dict(category_counts["greedy"]),
        "by_split_guided_categories": {
            k: dict(v) for k, v in category_counts.items() if k.startswith("guided_")
        },
    }

    return results, summary


def generate_markdown_report(
    clean_summary: Dict[str, Any],
    noisy_summary: Dict[str, Any],
    clean_results: Sequence[ThreeWayComparisonResult],
    noisy_results: Sequence[ThreeWayComparisonResult],
) -> str:
    """
    Constructs a comprehensive PR-ready markdown breakdown report.
    """
    lines: List[str] = []
    lines.append(
        "# Comparative Error Breakdown: Ground Truth vs. Greedy ASR vs. Syllabary-Guided CTC"
    )
    lines.append("")
    lines.append(
        "Comprehensive error categorization across 1,389 Cherokee Syllabary dataset samples post lateral deaffrication and laryngeal masking reforms."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & 3-Way Benchmark Comparison")
    lines.append("")
    lines.append("### A. Clean Audio Condition")
    lines.append("")
    lines.append(
        "| Split | Total Samples | Exact Match (Both) | Exact Match (Guided Only) | Exact Match (Greedy Only) | Guided CER Wins | Greedy CER Wins | Equal CER |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for s_name in ["test", "valid", "train", "all"]:
        st = clean_summary["split_stats"].get(s_name, {})
        tot = st.get("total", 0)
        be = st.get("both_exact", 0)
        go = st.get("guided_only_exact", 0)
        gro = st.get("greedy_only_exact", 0)
        gw = st.get("guided_better_cer", 0)
        grw = st.get("greedy_better_cer", 0)
        eq = st.get("equal_cer", 0)
        lines.append(
            f"| **{s_name}** | {tot} | {be} ({be/tot*100:.1f}%) | {go} ({go/tot*100:.1f}%) | {gro} ({gro/tot*100:.1f}%) | {gw} ({gw/tot*100:.1f}%) | {grw} ({grw/tot*100:.1f}%) | {eq} ({eq/tot*100:.1f}%) |"
        )

    lines.append("")
    lines.append("### B. Noisy Audio Condition (Pink Noise 18 dB SNR)")
    lines.append("")
    lines.append(
        "| Split | Total Samples | Exact Match (Both) | Exact Match (Guided Only) | Exact Match (Greedy Only) | Guided CER Wins | Greedy CER Wins | Equal CER |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for s_name in ["test", "valid", "train", "all"]:
        st = noisy_summary["split_stats"].get(s_name, {})
        tot = st.get("total", 0)
        be = st.get("both_exact", 0)
        go = st.get("guided_only_exact", 0)
        gro = st.get("greedy_only_exact", 0)
        gw = st.get("guided_better_cer", 0)
        grw = st.get("greedy_better_cer", 0)
        eq = st.get("equal_cer", 0)
        lines.append(
            f"| **{s_name}** | {tot} | {be} ({be/tot*100:.1f}%) | **{go} ({go/tot*100:.1f}%)** | {gro} ({gro/tot*100:.1f}%) | **{gw} ({gw/tot*100:.1f}%)** | {grw} ({grw/tot*100:.1f}%) | {eq} ({eq/tot*100:.1f}%) |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Syllabary-Guided Error Taxonomy & Event Breakdown")
    lines.append("")
    lines.append(
        "Detailed event counts for all residual degradation types in Syllabary-Guided CTC Segmentation:"
    )
    lines.append("")
    lines.append(
        "| Error Category | Clean Events (All) | Clean Events (Test+Valid) | Noisy Events (All) | Noisy Events (Test+Valid) | Primary Linguistic & Structural Cause |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    all_cats = sorted(
        set(
            list(clean_summary["guided_error_categories"].keys())
            + list(noisy_summary["guided_error_categories"].keys())
        )
    )

    clean_tv = Counter()
    for s_name in ["test", "valid"]:
        clean_tv.update(
            clean_summary.get("by_split_guided_categories", {}).get(
                f"guided_{s_name}", {}
            )
        )
    noisy_tv = Counter()
    for s_name in ["test", "valid"]:
        noisy_tv.update(
            noisy_summary.get("by_split_guided_categories", {}).get(
                f"guided_{s_name}", {}
            )
        )

    cause_map = {
        "LARYNGEAL_STRIDE_VOWEL_INTRUSION": "Syllable vowel retained alongside intrusive h in trellis (e.g. `akhahth` vs `akhth`, `kalhih` vs `kalh`)",
        "VOWEL_UNDER_SYNCOPE_MEDIAL": "Speaker syncopated medial vowel, but aligner emitted template vowel (e.g. `tsunahsti` vs `tsunhsti`)",
        "INITIAL_SIBILANT_HS_VS_S": "Template maps initial Syllabary Ꮝ to `hs-`, but GT transcript uses `s-` (e.g. `hstaya` vs `staya`)",
        "SONORANT_PREASPIRATION": "Pre-aspiration / devoicing mismatch on sonorants `nh, wh, yh, lh` (e.g. `uwhtohti` vs `uwtohti`)",
        "VOWEL_OVER_SYNCOPE_MEDIAL": "Speaker pronounced medial vowel, but aligner syncopated (e.g. `totalv` -> `ttalv`)",
        "GLOTTAL_STOP_HIATUS": "Presence/absence of hiatus glottal stop apostrophe `'` in GT vs Syllabary",
        "STOP_AFFRICATE_ASPIRATION": "Voicing/aspiration mismatch on stops & affricates (`th/t`, `kh/k`, `tsh/ts`)",
        "VOWEL_OVER_SYNCOPE_FINAL": "Final vowel dropped by aligner before boundary",
        "VOWEL_UNDER_SYNCOPE_INITIAL": "Initial onset vowel retained from template when speaker syncopated",
        "VOWEL_UNDER_SYNCOPE_FINAL": "Word-final vowel retained from template when speaker syncopated",
        "MEDIAL_SIBILANT_PREASPIRATION": "Medial preconsonantal sibilant aspiration `hs` vs `s`",
        "ISOLATED_LARYNGEAL_H": "Intervocalic / isolated stray `h`",
        "TRANSCRIPT_GT_DISCORDANCE": "Ground truth transcript divergence from spoken audio / Syllabary",
        "RESIDUAL_SUBSTITUTION": "Single-character phoneme substitutions",
    }

    for cat in sorted(
        all_cats,
        key=lambda c: clean_summary["guided_error_categories"].get(c, 0),
        reverse=True,
    ):
        c_all = clean_summary["guided_error_categories"].get(cat, 0)
        c_tv = clean_tv.get(cat, 0)
        n_all = noisy_summary["guided_error_categories"].get(cat, 0)
        n_tv = noisy_tv.get(cat, 0)
        cause = cause_map.get(cat, "Phonetic / orthographic discrepancy")
        lines.append(f"| **`{cat}`** | {c_all} | {c_tv} | {n_all} | {n_tv} | {cause} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed Case Studies & Representative Examples")
    lines.append("")

    # Collect representative examples for top categories (prioritize test and valid splits)
    examples_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_samples_by_cat: Dict[str, set] = defaultdict(set)
    # Sort results prioritizing test and valid
    sorted_results = sorted(
        clean_results,
        key=lambda r: 0 if r.split == "test" else (1 if r.split == "valid" else 2),
    )
    for res in sorted_results:
        for ev in res.guided_events:
            if (
                len(examples_by_cat[ev.category]) < 3
                and res.sample_id not in seen_samples_by_cat[ev.category]
            ):
                seen_samples_by_cat[ev.category].add(res.sample_id)
                examples_by_cat[ev.category].append(
                    {
                        "sample_id": res.sample_id,
                        "split": res.split,
                        "template": res.template,
                        "ground_truth": res.ground_truth,
                        "guided_hyp": res.guided_hyp,
                        "greedy_hyp": res.greedy_hyp,
                        "diff": f"GT: '{ev.gt_chunk}' -> Guided: '{ev.hyp_chunk}' ({ev.description})",
                    }
                )

    for cat in [
        "LARYNGEAL_STRIDE_VOWEL_INTRUSION",
        "VOWEL_UNDER_SYNCOPE_MEDIAL",
        "INITIAL_SIBILANT_HS_VS_S",
        "SONORANT_PREASPIRATION",
        "VOWEL_OVER_SYNCOPE_MEDIAL",
        "GLOTTAL_STOP_HIATUS",
    ]:
        exs = examples_by_cat.get(cat, [])
        if not exs:
            continue
        lines.append(f"### Category: `{cat}`")
        lines.append("")
        for ex in exs:
            lines.append(f"- **Sample ID**: `{ex['sample_id']}` (`{ex['split']}`)")
            lines.append(f"  - **Template**: `{ex['template']}`")
            lines.append(f"  - **Ground Truth**: `{ex['ground_truth']}`")
            lines.append(f"  - **Guided Hyp**:   `{ex['guided_hyp']}`")
            lines.append(f"  - **Greedy Hyp**:   `{ex['greedy_hyp']}`")
            lines.append(f"  - **Event**: {ex['diff']}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Targeted Next Interventions")
    lines.append("")
    lines.append(
        "1. **Laryngeal Stride Mutex Constraint (`LARYNGEAL_STRIDE_VOWEL_INTRUSION`, 210 clean events)**:"
    )
    lines.append(
        "   - In `prepare_cherokee_text`, enforce a strict mutex in syllabic trellis expansion: a syllable containing an intrusive `h` and an optional syncope vowel must branch into *either* `[Consonant + h + NextConsonant]` (aspirated syncope) *or* `[Consonant + Vowel]` (full syllable), never emitting both `h` and `V` in tandem (`k-h-a-h-th`)."
    )
    lines.append(
        "2. **Word-Initial Sibilant Normalizer (`INITIAL_SIBILANT_HS_VS_S`, 93 events)**:"
    )
    lines.append(
        "   - Standardize Syllabary `Ꮝ` at word boundary: allow both `s` and `hs` in base template normalizer or evaluate with canonicalized sibilant onsets."
    )
    lines.append(
        "3. **Sonorant Pre-Aspiration / Devoicing Licensing (`SONORANT_PREASPIRATION`, 87 clean / 194 noisy events)**:"
    )
    lines.append(
        "   - Broaden trellis phonotactics for pre-vocalic and intervocalic sonorants (`n, w, y, l`) to license optional `nh, wh, yh, lh` paths dynamically."
    )
    lines.append(
        "4. **Medial Vocalic Syncope Weighting (`VOWEL_UNDER_SYNCOPE_MEDIAL`, 135 clean events)**:"
    )
    lines.append(
        "   - Calibrate relative contrastive gating or phonotactic penalty for medial short vowels (`a, i, v`) preceding voiceless consonants."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze 3-way comparative errors across Cherokee Syllabary dataset."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=BASE_DIR / "runs" / "evaluation" / "rescore_syllabary_results.json",
        help="Path to rescore_syllabary_results.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "runs" / "evaluation" / "comparative_error_analysis.json",
        help="Path to save comparative error analysis JSON",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=BASE_DIR / "runs" / "evaluation" / "comparative_error_report.md",
        help="Path to save markdown error breakdown report",
    )

    args = parser.parse_args()

    if not args.input_json.exists():
        logger.error("Input JSON results file not found at: %s", args.input_json)
        sys.exit(1)

    logger.info("Loading evaluation results from: %s", args.input_json)
    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    clean_samples = payload.get("clean_samples", [])
    noisy_samples = payload.get("noisy_samples", [])

    logger.info("Analyzing clean samples (%d entries)...", len(clean_samples))
    clean_results, clean_summary = analyze_dataset_condition(clean_samples, "clean")

    logger.info("Analyzing noisy samples (%d entries)...", len(noisy_samples))
    noisy_results, noisy_summary = analyze_dataset_condition(noisy_samples, "noisy")

    # Generate Markdown Report
    report_md = generate_markdown_report(
        clean_summary=clean_summary,
        noisy_summary=noisy_summary,
        clean_results=clean_results,
        noisy_results=noisy_results,
    )

    # Save Output JSON
    out_payload = {
        "timestamp": payload.get("timestamp", ""),
        "model_name": payload.get("model_name", ""),
        "clean_summary": clean_summary,
        "noisy_summary": noisy_summary,
        "clean_results": [r.to_dict() for r in clean_results],
        "noisy_results": [r.to_dict() for r in noisy_results],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved comparative error analysis JSON to: %s", args.output_json)

    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Saved markdown breakdown report to: %s", args.output_report)

    # Print Summary to Terminal
    print("\n" + "=" * 96)
    print(f"{'3-WAY COMPARATIVE ERROR BREAKDOWN SUMMARY':^96}")
    print("=" * 96)
    print(
        f"{'Condition':<8} | {'Category':<35} | {'All Count':<10} | {'Test+Valid':<12} | {'Description':<25}"
    )
    print("-" * 96)

    for cond_name, summary in [("Clean", clean_summary), ("Noisy", noisy_summary)]:
        cats = summary["guided_error_categories"]
        tv_counter = Counter()
        for s_name in ["test", "valid"]:
            tv_counter.update(
                summary.get("by_split_guided_categories", {}).get(
                    f"guided_{s_name}", {}
                )
            )

        for cat, cnt in sorted(cats.items(), key=lambda x: x[1], reverse=True)[:8]:
            tv_cnt = tv_counter.get(cat, 0)
            print(f"{cond_name:<8} | {cat:<35} | {cnt:>9d} | {tv_cnt:>11d} |")
        print("-" * 96)

    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
