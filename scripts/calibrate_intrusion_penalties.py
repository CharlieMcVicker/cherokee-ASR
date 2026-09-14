#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_intrusion_penalties.py

Runs grid search across /h/ and /'/ intrusion penalties, min-logprobs, and syncope penalties
on Mark Chapter 1 using cached acoustic emissions, logging diff categories, glottal recovery,
aspiration suppression, and anomaly flags.
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.models import CTCAlignerConfig, TextChunk
from transcription.models.asr_model import CherokeeASRModel
from transcription.new_testament.pipeline import load_chapter_transcript

DEFAULT_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
DEFAULT_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"


def run_grid_search(
    book: str = "mark",
    chapter: int = 1,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    print(f"Loading model {model_repo} (revision: {model_revision}) ...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN", None)
    asr_model = CherokeeASRModel.from_pretrained_or_best(
        path_or_repo=model_repo,
        revision=model_revision,
        device=device,
        token=token,
    )

    audio_path = (
        BASE_DIR
        / "cherokee_new_testament"
        / "audio_source"
        / f"{book}_{chapter:02d}.mp3"
    )
    transcript_path = (
        BASE_DIR
        / "cherokee_new_testament"
        / "book_transcripts"
        / f"{book}_{chapter:02d}.json"
    )

    transcript_data = load_chapter_transcript(transcript_path)
    chunks = [
        TextChunk(chunk_id=verse_id, text=verse_info.get("phonetic", ""))
        for verse_id, verse_info in transcript_data.items()
    ]

    # Pre-extract/ensure logits are cached
    base_aligner = CTCSegmentationAligner(
        model=asr_model,
        config=CTCAlignerConfig(cache=True, cache_dir=cache_dir),
    )
    lpz, dur_sec, _ = base_aligner.get_logits_cached(
        audio_path, asr_model=asr_model, cache=True
    )
    print(
        f"Loaded cached logits for {book}_{chapter:02d} (shape: {lpz.shape}, dur: {dur_sec:.2f}s)"
    )

    # Grid candidate configurations
    h_penalties = [3.0, 4.0, 4.5, 5.0]
    glottal_penalties = [0.5, 0.8, 1.0, 1.2]
    min_logprob_options = [
        None,
        {"h": float(np.log(0.35)), "'": float(np.log(0.20))},
    ]

    grid_results = []

    for h_p in h_penalties:
        for g_p in glottal_penalties:
            for mlp in min_logprob_options:
                penalties = {"h": h_p, "'": g_p}
                mlp_str = "with_mlp" if mlp is not None else "no_mlp"
                cfg_name = f"h_{h_p}_glottal_{g_p}_{mlp_str}"
                print(f"Testing configuration: {cfg_name} ...")

                aligner = CTCSegmentationAligner(
                    model=asr_model,
                    config=CTCAlignerConfig(
                        cache=True,
                        cache_dir=cache_dir,
                        syncope_penalty=6.0,
                        intrusive_penalties=penalties,
                        intrusive_min_logprobs=mlp,
                        flag_min_char_confidence=0.005,
                        enforce_phonotactics=True,
                    ),
                )

                t0 = time.time()
                res = aligner.align(audio_path, chunks=chunks, cache=True)
                elapsed = time.time() - t0

                # Analyze emitted text across chapter
                total_glottal_stops = 0
                total_h_aspirations = 0
                total_flagged_words = 0
                anomalous_verses = []
                mark_1_1_flagged = False

                for c in res.aligned_chunks:
                    emitted = c.emitted_text or ""
                    total_glottal_stops += emitted.count("'")
                    # Count 'h' not part of th, kh, tsh, tlh, ch, sh
                    total_h_aspirations += emitted.count("h")

                    if c.chunk_id == "020101":
                        for w in c.words:
                            if w.word == "yihstv" and w.flagged:
                                mark_1_1_flagged = True

                    for w in c.words:
                        if w.flagged:
                            total_flagged_words += 1

                    if c.has_anomalies:
                        anomalous_verses.append(c.chunk_id)

                summary = {
                    "config_name": cfg_name,
                    "h_penalty": h_p,
                    "glottal_penalty": g_p,
                    "min_logprobs": mlp,
                    "total_glottal_stops": total_glottal_stops,
                    "total_h_aspirations": total_h_aspirations,
                    "total_flagged_words": total_flagged_words,
                    "anomalous_verses_count": len(anomalous_verses),
                    "mark_1_1_yihstv_flagged": mark_1_1_flagged,
                    "mean_distance_score": (
                        res.metrics.mean_distance_score if res.metrics else 0.0
                    ),
                    "elapsed_sec": round(elapsed, 2),
                }
                grid_results.append(summary)
                print(
                    f"  -> Glottals: {total_glottal_stops}, Aspirations: {total_h_aspirations}, "
                    f"Flagged: {total_flagged_words}, Mark1:1 Flagged: {mark_1_1_flagged}"
                )

    out_dir = BASE_DIR / "runs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"calibration_{book}_{chapter:02d}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)

    print(f"\nSaved calibration grid search results to {out_file}")
    return grid_results


if __name__ == "__main__":
    run_grid_search()
