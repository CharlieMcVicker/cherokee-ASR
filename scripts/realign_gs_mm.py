# -*- coding: utf-8 -*-
"""
scripts/realign_gs_mm.py

Realigns the saving-the-voices/gs_mm interview audio and transcript using the
code-switched greedy aligner with calibrated confusion matrix projection.

CRITICAL MODEL SELECTION:
- Toneless Pre-Bible Model:
  - Repo: charliemcvicker/length-only-20260704-155307-asr-cherokee-colon
  - Revision: 76e62140955f4738abdab345ea34068b02d8d2a2
  - Device: mps (or cpu fallback)
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Set

import torch

from digohwelisgi.cherokee.codeswitching import get_default_projector
from digohwelisgi.alignment.pipeline import align_syllabary_greedy
from digohwelisgi.cherokee.models import CherokeeASRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PRE_BIBLE_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
PRE_BIBLE_MODEL_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"

AUDIO_PATH = Path("data/projects/saving-the-voices/gs_mm.wav")
TRANSCRIPT_PATH = Path("data/projects/saving-the-voices/gs_mm.txt")
OUTPUT_DIR = Path("data/projects/saving-the-voices/output_codeswitched")
BASELINE_DIR = Path("data/projects/saving-the-voices/output_greedy")

TEXTGRID_FILENAME = "gs_mm_codeswitched.TextGrid"
MANIFEST_FILENAME = "gs_mm_codeswitched_manifest.json"

AUDIT_KEYWORDS = {
    "guy",
    "soldier",
    "jay",
    "charley",
    "mccoy",
    "dry",
    "creek",
    "yeah",
    "ok",
}


def run_interview_realignment() -> Dict[str, Any]:
    start_time = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading toneless pre-Bible model on device: {device}...")

    model = CherokeeASRModel.from_pretrained(
        path_or_repo=PRE_BIBLE_MODEL_REPO,
        revision=PRE_BIBLE_MODEL_REVISION,
        device=device,
    )
    logger.info("Model loaded successfully.")

    logger.info("Initializing calibrated synthetic target projector...")
    projector = get_default_projector()
    logger.info("Projector initialized.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting code-switched greedy alignment for {AUDIO_PATH} -> {OUTPUT_DIR}..."
    )
    alignment_output = align_syllabary_greedy(
        audio=AUDIO_PATH,
        transcript=TRANSCRIPT_PATH,
        output_dir=OUTPUT_DIR,
        model=model,
        projector=projector,
        code_switched=True,
        reconcile=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename=TEXTGRID_FILENAME,
        manifest_filename=MANIFEST_FILENAME,
    )
    elapsed_sec = time.time() - start_time
    logger.info(f"Alignment completed in {elapsed_sec:.2f}s.")

    # Compare against baseline
    baseline_manifest_path = BASELINE_DIR / "gs_mm_greedy_manifest.json"
    codeswitched_manifest_path = OUTPUT_DIR / MANIFEST_FILENAME

    with open(codeswitched_manifest_path, "r", encoding="utf-8") as f:
        cs_manifest = json.load(f)

    baseline_metrics = {}
    if baseline_manifest_path.exists():
        with open(baseline_manifest_path, "r", encoding="utf-8") as f:
            base_manifest = json.load(f)
            baseline_metrics = base_manifest.get("metrics", {})

    cs_metrics = cs_manifest.get("metrics", {})

    logger.info("=" * 60)
    logger.info("ALIGNMENT METRICS COMPARISON (BASELINE vs CODE-SWITCHED):")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} | {'Baseline':<15} | {'Code-Switched':<15}")
    logger.info("-" * 66)
    for k in [
        "total_chunks",
        "matched_chunks",
        "match_ratio",
        "mean_distance_score",
        "total_ground_truth_chars",
        "total_emitted_chars",
    ]:
        b_val = baseline_metrics.get(k, "N/A")
        c_val = cs_metrics.get(k, "N/A")
        if isinstance(b_val, float):
            b_val = f"{b_val:.4f}"
        if isinstance(c_val, float):
            c_val = f"{c_val:.4f}"
        logger.info(f"{k:<30} | {str(b_val):<15} | {str(c_val):<15}")
    logger.info("=" * 60)

    # Keyword Audit
    logger.info("AUDITING KEY CODE-SWITCHED WORDS...")
    keyword_audit_results: Dict[str, List[Dict[str, Any]]] = {
        kw: [] for kw in AUDIT_KEYWORDS
    }

    for line in cs_manifest.get("lines", []):
        line_id = line.get("line_id", "")
        # Inspect words and additional word tiers
        words = line.get("words", [])
        add_tiers = line.get("additional_word_tiers", {})
        eng_tier = add_tiers.get("English Words", [])
        syll_tier = add_tiers.get("Syllabary Words", [])

        for w_idx, w in enumerate(words):
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            dur = end - start

            # Check English Words tier or word tokens
            eng_word = ""
            if w_idx < len(eng_tier):
                eng_word = eng_tier[w_idx].get("word", "")

            syll_word = ""
            if w_idx < len(syll_tier):
                syll_word = syll_tier[w_idx].get("word", "")

            tokens_to_check = set()
            if eng_word:
                tokens_to_check.update(eng_word.lower().split())
            if syll_word:
                tokens_to_check.update(syll_word.lower().split())

            for tok in tokens_to_check:
                cleaned_tok = tok.strip(".,?!:;\"'")
                if cleaned_tok in AUDIT_KEYWORDS:
                    keyword_audit_results[cleaned_tok].append(
                        {
                            "line_id": line_id,
                            "word": cleaned_tok,
                            "display_word": eng_word or syll_word,
                            "canonical_tth": w.get("word", ""),
                            "start": start,
                            "end": end,
                            "duration": dur,
                            "confidence": w.get("confidence", 0.0),
                            "emitted_word": w.get("emitted_word", ""),
                        }
                    )

    logger.info(
        f"{'Keyword':<12} | {'Count':<6} | {'Valid Bounds':<12} | {'Avg Dur (s)':<12} | {'Sample Timestamp'}"
    )
    logger.info("-" * 75)
    for kw in sorted(AUDIT_KEYWORDS):
        occurrences = keyword_audit_results[kw]
        count = len(occurrences)
        valid_bounds = (
            all(o["duration"] > 0 for o in occurrences) if count > 0 else False
        )
        avg_dur = sum(o["duration"] for o in occurrences) / count if count > 0 else 0.0
        sample_ts = (
            f"{occurrences[0]['start']:.2f}s - {occurrences[0]['end']:.2f}s ({occurrences[0]['line_id']})"
            if count > 0
            else "N/A"
        )
        logger.info(
            f"{kw:<12} | {count:<6} | {str(valid_bounds):<12} | {avg_dur:<12.3f} | {sample_ts}"
        )
    logger.info("=" * 75)

    return {
        "metrics": cs_metrics,
        "baseline_metrics": baseline_metrics,
        "keyword_audit": keyword_audit_results,
    }


if __name__ == "__main__":
    run_interview_realignment()
