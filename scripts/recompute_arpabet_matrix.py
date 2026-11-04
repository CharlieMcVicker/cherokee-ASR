#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/recompute_arpabet_matrix.py

Recomputes the ARPAbet-to-Cherokee acoustic confusion matrix using the newly
promoted best model (checkpoint-8000), compares it against the pre-Bible baseline,
and generates comparative visualization plots.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from digohwelisgi.cherokee.models import CherokeeASRModel
from digohwelisgi.utils.model_utils import get_best_model_config
from digohwelisgi.cherokee.phonotactics import tokenize_phonemes
from digohwelisgi.cherokee.codeswitching.types import (
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
)
from digohwelisgi.core.codeswitching.matrix import (
    AcousticConfusionMatrix,
    normalize_source_key,
    normalize_target_key,
)
from digohwelisgi.core.codeswitching.trainer import (
    ConfusionMatrixTrainer,
    sanitize_model_id,
)
from digohwelisgi.core.codeswitching.types import (
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
)
from digohwelisgi.core.models import infer_emissions_batch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recompute_arpabet")

# Explicit place and manner distances between ARPAbet and Cherokee phonemes.
ARTICULATORY_FEATURE_DISTANCES: Dict[str, Dict[str, float]] = {
    "AA": {"a": 0.2, "o": 0.5, "v": 0.7, "e": 1.2, "u": 1.4, "i": 1.6},
    "AE": {"a": 0.2, "e": 0.4, "v": 0.7, "i": 1.0, "o": 1.5, "u": 1.7},
    "AH": {"a": 0.2, "v": 0.3, "o": 0.5, "e": 0.6, "u": 1.0, "i": 1.2},
    "AO": {"o": 0.2, "a": 0.4, "u": 0.5, "v": 0.7, "e": 1.4, "i": 1.6},
    "AW": {
        "aw": 0.2,
        "au": 0.2,
        "av": 0.3,
        "a": 0.3,
        "u": 0.4,
        "o": 0.5,
        "w": 0.6,
        "v": 0.7,
        "e": 1.4,
        "i": 1.5,
    },
    "AY": {
        "ai": 0.2,
        "ay": 0.2,
        "a": 0.3,
        "i": 0.4,
        "e": 0.4,
        "y": 0.6,
        "v": 0.8,
        "o": 1.4,
        "u": 1.6,
    },
    "EH": {"e": 0.2, "a": 0.4, "i": 0.6, "v": 0.8, "o": 1.4, "u": 1.6},
    "ER": {"v": 0.3, "e": 0.4, "a": 0.5, "l": 0.8, "o": 1.0, "u": 1.2, "i": 1.2},
    "EY": {
        "ei": 0.2,
        "ey": 0.2,
        "e": 0.2,
        "i": 0.4,
        "a": 0.6,
        "v": 0.9,
        "o": 1.5,
        "u": 1.7,
    },
    "IH": {"i": 0.2, "e": 0.4, "a": 0.8, "v": 0.9, "u": 1.2, "o": 1.5},
    "IY": {"i": 0.2, "e": 0.5, "y": 0.6, "a": 1.2, "v": 1.4, "u": 1.5, "o": 1.8},
    "OW": {
        "ou": 0.2,
        "ow": 0.2,
        "o": 0.2,
        "u": 0.4,
        "w": 0.6,
        "a": 0.8,
        "v": 0.9,
        "e": 1.5,
        "i": 1.8,
    },
    "OY": {
        "oi": 0.2,
        "oy": 0.2,
        "o": 0.3,
        "i": 0.4,
        "e": 0.5,
        "u": 0.7,
        "a": 0.8,
        "y": 0.6,
    },
    "UH": {"u": 0.2, "o": 0.4, "v": 0.6, "a": 0.9, "e": 1.4, "i": 1.5},
    "UW": {"u": 0.2, "o": 0.4, "w": 0.5, "v": 0.7, "a": 1.2, "e": 1.5, "i": 1.8},
    "B": {"w": 0.3, "wh": 0.4, "t": 0.5, "k": 0.5, "th": 0.6, "kh": 0.6, "m": 0.6},
    "D": {"t": 0.2, "th": 0.4, "ts": 0.5, "tsh": 0.6, "k": 0.8, "kh": 0.9},
    "G": {"k": 0.2, "kw": 0.3, "kh": 0.4, "kwh": 0.5, "t": 0.8, "th": 0.9},
    "P": {
        "th": 0.3,
        "t": 0.4,
        "wh": 0.4,
        "kh": 0.4,
        "k": 0.5,
        "w": 0.5,
        "kwh": 0.5,
        "kw": 0.6,
    },
    "T": {
        "th": 0.2,
        "t": 0.3,
        "ts": 0.5,
        "tsh": 0.5,
        "s": 0.6,
        "hs": 0.6,
        "kh": 0.8,
        "k": 0.9,
    },
    "K": {
        "kh": 0.2,
        "k": 0.3,
        "kwh": 0.4,
        "kw": 0.4,
        "th": 0.8,
        "t": 0.9,
    },
    "CH": {"tsh": 0.2, "ts": 0.3, "s": 0.5, "hs": 0.5, "th": 0.7},
    "JH": {"ts": 0.2, "tsh": 0.4, "s": 0.5, "hs": 0.5, "t": 0.7},
    "S": {"s": 0.2, "hs": 0.3, "ts": 0.5, "tsh": 0.6, "th": 0.8},
    "Z": {"s": 0.2, "hs": 0.3, "ts": 0.4, "tsh": 0.5},
    "SH": {"s": 0.2, "hs": 0.3, "ts": 0.4, "tsh": 0.4},
    "ZH": {"s": 0.3, "hs": 0.4, "ts": 0.3, "tsh": 0.4},
    "TH": {"th": 0.2, "t": 0.3, "s": 0.4, "hs": 0.4, "tsh": 0.6},
    "DH": {"t": 0.2, "th": 0.3, "s": 0.4, "hs": 0.4, "ts": 0.5},
    "F": {
        "wh": 0.3,
        "w": 0.4,
        "h": 0.4,
        "hs": 0.5,
        "th": 0.5,
        "s": 0.6,
        "kh": 0.6,
    },
    "V": {"w": 0.3, "wh": 0.4, "t": 0.6, "k": 0.6, "m": 0.6},
    "M": {"m": 0.2, "n": 0.5, "nh": 0.6, "w": 0.7},
    "N": {"n": 0.2, "nh": 0.4, "m": 0.5, "l": 0.7},
    "NG": {"nv": 0.2, "nk": 0.3, "n": 0.3, "nh": 0.4, "k": 0.5, "kh": 0.6, "m": 0.6},
    "L": {"l": 0.2, "lh": 0.3, "tl": 0.4, "tlh": 0.5, "w": 0.7},
    "R": {"l": 0.3, "lh": 0.4, "w": 0.4, "wh": 0.5, "tl": 0.6, "tlh": 0.7},
    "W": {"w": 0.2, "wh": 0.3, "kw": 0.4, "kwh": 0.5, "m": 0.7},
    "Y": {"y": 0.2, "yh": 0.3, "ts": 0.6, "i": 0.8},
    "HH": {"h": 0.2, "hs": 0.3, "'": 0.4, "wh": 0.5, "yh": 0.5},
    "S T": {"hst": 0.2, "st": 0.3, "t": 0.5, "s": 0.5},
    "S K": {"hsk": 0.2, "hskw": 0.25, "sk": 0.3, "kw": 0.4},
    "S P": {"hskw": 0.25, "hsp": 0.3, "kw": 0.4, "wh": 0.4},
    "T SH": {"tsh": 0.2, "ts": 0.3},
    "D ZH": {"ts": 0.2, "tsh": 0.4},
    "N G": {"nv": 0.2, "nk": 0.3, "n": 0.3},
}


def get_articulatory_distance(arpabet_phone: str, cherokee_phone: str) -> float:
    a = arpabet_phone.strip().upper()
    c = cherokee_phone.strip().lower()
    if a in ARTICULATORY_FEATURE_DISTANCES and c in ARTICULATORY_FEATURE_DISTANCES[a]:
        return ARTICULATORY_FEATURE_DISTANCES[a][c]
    tokens_a = a.split()
    if len(tokens_a) == 2 and len(c) >= 2:
        c1, c2 = c[:1], c[1:]
        d1 = get_articulatory_distance(tokens_a[0], c1)
        d2 = get_articulatory_distance(tokens_a[1], c2)
        return (d1 + d2) / 2.0
    is_a_vowel = a in STANDARD_ARPABET_VOWELS
    is_c_vowel = c in CANONICAL_CHEROKEE_VOWELS
    if is_a_vowel != is_c_vowel:
        return 2.4
    return 1.5 if is_a_vowel else 1.8


def build_articulatory_seed_matrix(
    arpabet_phonemes: Optional[Sequence[str]] = None,
    cherokee_phonemes: Optional[Sequence[str]] = None,
    model_id: str = "articulatory_seed",
    temperature: float = 0.5,
) -> AcousticConfusionMatrix:
    arp_vocab: Tuple[str, ...] = (
        tuple(arpabet_phonemes)
        if arpabet_phonemes is not None
        else STANDARD_ARPABET_PHONEMES
    )
    chr_vocab: Tuple[str, ...] = (
        tuple(cherokee_phonemes)
        if cherokee_phonemes is not None
        else CANONICAL_CHEROKEE_PHONEMES
    )

    probabilities: Dict[str, Dict[str, float]] = {}
    for a in arp_vocab:
        weights: Dict[str, float] = {}
        for c in chr_vocab:
            dist = get_articulatory_distance(a, c)
            weights[c] = math.exp(-dist / max(temperature, 1e-4))
        sum_w = sum(weights.values())
        if sum_w > 0:
            probabilities[a] = {c: w / sum_w for c, w in weights.items()}

    for k, dist_map in ARTICULATORY_FEATURE_DISTANCES.items():
        if k not in probabilities:
            weights = {
                c: math.exp(-d / max(temperature, 1e-4)) for c, d in dist_map.items()
            }
            sum_w = sum(weights.values())
            if sum_w > 0:
                probabilities[k] = {c: w / sum_w for c, w in weights.items()}

    ins_probs = {c: 0.05 for c in chr_vocab}
    del_probs = {a: 0.05 for a in arp_vocab}

    return AcousticConfusionMatrix.create(
        model_id=model_id,
        probabilities=probabilities,
        insertion_probabilities=ins_probs,
        deletion_probabilities=del_probs,
        arpabet_vocab=arp_vocab,
        cherokee_vocab=chr_vocab,
        prune_threshold=0.01,
        iteration=0,
    )


def run_emissions_and_caching(
    manifest_entries: List[Dict[str, Any]],
    model: CherokeeASRModel,
    cache_path: Path,
    model_id: str,
    batch_size: int = 32,
    device: str = "mps",
) -> List[Dict[str, Any]]:
    if cache_path.exists():
        logger.info(f"Loading existing emissions cache from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", [])

    logger.info(
        f"Computing emissions for {len(manifest_entries)} word clips on {device}..."
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    audio_paths = [e["audio_path"] for e in manifest_entries]
    t0 = time.time()

    # Sort by duration to minimize padding overhead
    indexed_paths = list(enumerate(audio_paths))
    indexed_paths.sort(key=lambda x: manifest_entries[x[0]].get("duration", 0.5))

    sorted_indices = [x[0] for x in indexed_paths]
    sorted_paths = [x[1] for x in indexed_paths]

    emissions_outputs = infer_emissions_batch(
        model=model.model,
        processor=model.processor,
        audio_inputs=sorted_paths,
        device=device,
        batch_size=batch_size,
    )

    dt = time.time() - t0
    logger.info(
        f"Inference complete in {dt:.1f}s ({len(audio_paths)/dt:.1f} clips/sec)."
    )

    # Map outputs back to original order
    reordered_outputs = [None] * len(manifest_entries)
    for orig_idx, out in zip(sorted_indices, emissions_outputs):
        reordered_outputs[orig_idx] = out

    cached_entries: List[Dict[str, Any]] = []
    for entry, out in zip(manifest_entries, reordered_outputs):
        assert out is not None
        greedy_text = out.decode_greedy()
        tokens = [t.symbol for t in tokenize_phonemes(greedy_text)]
        # Average token confidences if available
        confs = [0.95] * len(tokens)
        cached_entries.append(
            {
                "clip_id": entry["clip_id"],
                "word": entry["word"],
                "arpabet": [p["phone"] for p in entry["arpabet"]],
                "greedy_text": greedy_text,
                "tokens": tokens,
                "token_confidences": confs,
                "duration": entry.get("duration", 0.0),
            }
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entries": cached_entries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved {len(cached_entries)} emissions entries to {cache_path}")
    return cached_entries


def generate_comparison_visualizations(
    matrix_pre_bible: AcousticConfusionMatrix,
    matrix_new: AcousticConfusionMatrix,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)

    # 1. Top Consonant Mappings Comparison
    consonants = [
        "P",
        "T",
        "K",
        "B",
        "D",
        "G",
        "CH",
        "JH",
        "S",
        "SH",
        "TH",
        "DH",
        "F",
        "V",
        "M",
        "N",
        "L",
        "R",
        "W",
        "Y",
        "HH",
    ]

    # Collect top-1 target & probability for Pre-Bible and New Model
    pre_targets = [matrix_pre_bible.best_target_for(c)[0] for c in consonants]
    pre_probs = [matrix_pre_bible.best_target_for(c)[1] for c in consonants]

    new_targets = [matrix_new.best_target_for(c)[0] for c in consonants]
    new_probs = [matrix_new.best_target_for(c)[1] for c in consonants]

    y_pos = np.arange(len(consonants))
    bar_width = 0.38

    ax1 = axes[0, 0]
    bars1 = ax1.barh(
        y_pos - bar_width / 2,
        pre_probs,
        height=bar_width,
        label="Pre-Bible Baseline",
        color="#4A90E2",
        alpha=0.85,
    )
    bars2 = ax1.barh(
        y_pos + bar_width / 2,
        new_probs,
        height=bar_width,
        label="Checkpoint-8000 (New)",
        color="#50E3C2",
        alpha=0.85,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(consonants, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Top Target Probability P(Cherokee | ARPAbet)", fontsize=10)
    ax1.set_title(
        "Consonant Mappings: Top Target Probability", fontsize=12, fontweight="bold"
    )
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Add labels on bars with target phoneme
    for i, (b1, b2, t1, t2) in enumerate(zip(bars1, bars2, pre_targets, new_targets)):
        ax1.text(
            b1.get_width() + 0.01,
            b1.get_y() + b1.get_height() / 2,
            f"→{t1}",
            va="center",
            fontsize=8,
            color="#2A60A0",
        )
        ax1.text(
            b2.get_width() + 0.01,
            b2.get_y() + b2.get_height() / 2,
            f"→{t2}",
            va="center",
            fontsize=8,
            color="#1B8065",
            fontweight="bold",
        )
    ax1.set_xlim(0, 1.18)

    # 2. Vowel Mappings Comparison
    vowels = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    ]
    v_y_pos = np.arange(len(vowels))

    pre_v_targets = [matrix_pre_bible.best_target_for(v)[0] for v in vowels]
    pre_v_probs = [matrix_pre_bible.best_target_for(v)[1] for v in vowels]

    new_v_targets = [matrix_new.best_target_for(v)[0] for v in vowels]
    new_v_probs = [matrix_new.best_target_for(v)[1] for v in vowels]

    ax2 = axes[0, 1]
    bars_v1 = ax2.barh(
        v_y_pos - bar_width / 2,
        pre_v_probs,
        height=bar_width,
        label="Pre-Bible Baseline",
        color="#E67E22",
        alpha=0.85,
    )
    bars_v2 = ax2.barh(
        v_y_pos + bar_width / 2,
        new_v_probs,
        height=bar_width,
        label="Checkpoint-8000 (New)",
        color="#9B59B6",
        alpha=0.85,
    )
    ax2.set_yticks(v_y_pos)
    ax2.set_yticklabels(vowels, fontsize=9, fontweight="bold")
    ax2.set_xlabel("Top Target Probability P(Cherokee | ARPAbet)", fontsize=10)
    ax2.set_title(
        "Vowel Mappings: Top Target Probability", fontsize=12, fontweight="bold"
    )
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (b1, b2, t1, t2) in enumerate(
        zip(bars_v1, bars_v2, pre_v_targets, new_v_targets)
    ):
        ax2.text(
            b1.get_width() + 0.01,
            b1.get_y() + b1.get_height() / 2,
            f"→{t1}",
            va="center",
            fontsize=8,
            color="#B05000",
        )
        ax2.text(
            b2.get_width() + 0.01,
            b2.get_y() + b2.get_height() / 2,
            f"→{t2}",
            va="center",
            fontsize=8,
            color="#6C2E85",
            fontweight="bold",
        )
    ax2.set_xlim(0, 1.18)

    # 3. Probability Delta (Confidence Shift)
    all_phones = consonants + vowels
    deltas = []
    for p in all_phones:
        p_old = matrix_pre_bible.best_target_for(p)[1]
        p_new = matrix_new.best_target_for(p)[1]
        deltas.append(p_new - p_old)

    ax3 = axes[1, 0]
    colors = ["#2ECC71" if d >= 0 else "#E74C3C" for d in deltas]
    bars_d = ax3.bar(np.arange(len(all_phones)), deltas, color=colors, alpha=0.85)
    ax3.set_xticks(np.arange(len(all_phones)))
    ax3.set_xticklabels(all_phones, rotation=90, fontsize=8)
    ax3.set_ylabel("Probability Shift (New - Old)", fontsize=10)
    ax3.set_title(
        "Top-Target Confidence Shift Across Phonemes", fontsize=12, fontweight="bold"
    )
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Summary Matrix Heatmap: Most Frequent Substitutions
    ax4 = axes[1, 1]
    common_chr = [
        "a",
        "e",
        "i",
        "o",
        "u",
        "v",
        "t",
        "th",
        "k",
        "kh",
        "s",
        "hs",
        "ts",
        "tsh",
        "w",
        "y",
        "l",
        "n",
        "m",
    ]
    subset_arp = [
        "P",
        "T",
        "K",
        "B",
        "D",
        "G",
        "S",
        "Z",
        "CH",
        "JH",
        "M",
        "N",
        "L",
        "R",
        "AA",
        "IY",
        "UW",
    ]

    heatmap_data = np.zeros((len(subset_arp), len(common_chr)))
    for r, arp in enumerate(subset_arp):
        for c, chr_p in enumerate(common_chr):
            heatmap_data[r, c] = matrix_new.get_probability(arp, chr_p)

    im = ax4.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")
    ax4.set_xticks(np.arange(len(common_chr)))
    ax4.set_yticks(np.arange(len(subset_arp)))
    ax4.set_xticklabels(common_chr, fontsize=9, fontweight="bold")
    ax4.set_yticklabels(subset_arp, fontsize=9, fontweight="bold")
    ax4.set_title(
        "Checkpoint-8000 Substitution Distribution (Heatmap)",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(
        im,
        ax=ax4,
        orientation="horizontal",
        pad=0.12,
        label="Conditional Probability P(Cherokee | ARPAbet)",
    )

    plt.suptitle(
        "ARPAbet-to-Cherokee Acoustic Alignment: Pre-Bible Baseline vs Checkpoint-8000",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved successfully to {output_png}")


def main() -> None:
    manifest_path = Path("data/arpabet_alignment/words_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_entries = json.load(f)
    logger.info(f"Loaded {len(manifest_entries)} words from {manifest_path}.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading best model (checkpoint-8000) on {device}...")
    best_config = get_best_model_config()
    repo = best_config["repo"]
    revision = best_config.get("revision")
    model = CherokeeASRModel.get_best_model(device=device)

    model_id = f"charliemcvicker_{repo.split('/')[-1]}_{revision or 'latest'}"
    cache_path = Path(f"data/arpabet_alignment/cache/{model_id}_emissions.json")
    matrix_output_path = Path(
        f"data/arpabet_alignment/matrices/{model_id}_confusion_matrix.json"
    )

    # 1. Forward inference & emissions
    emissions_entries = run_emissions_and_caching(
        manifest_entries=manifest_entries,
        model=model,
        cache_path=cache_path,
        model_id=model_id,
        batch_size=32,
        device=device,
    )

    # 2. Build paired training data
    paired_data: List[
        Tuple[Sequence[str], Sequence[str], Optional[Sequence[float]]]
    ] = []
    for entry in emissions_entries:
        arp_phones = entry["arpabet"]
        chr_tokens = entry["tokens"]
        confs = entry.get("token_confidences")
        if arp_phones and chr_tokens:
            paired_data.append((arp_phones, chr_tokens, confs))

    logger.info(f"Prepared {len(paired_data)} alignment pairs for EM estimation.")

    # 3. Build articulatory seed matrix
    logger.info("Initializing articulatory feature seed matrix...")
    seed_matrix = build_articulatory_seed_matrix(model_id="articulatory_seed")

    # 4. Train with EM
    logger.info("Starting Expectation-Maximization training (4 iterations)...")
    trainer = ConfusionMatrixTrainer(
        num_iterations=4, prune_threshold=0.05, alpha_prior=0.05
    )
    t_train = time.time()
    trained_matrix = trainer.train(paired_data, seed_matrix, model_id=model_id)
    logger.info(f"EM training completed in {time.time() - t_train:.2f}s.")

    # 5. Save matrix
    matrix_output_path.parent.mkdir(parents=True, exist_ok=True)
    trained_matrix.save(matrix_output_path)
    logger.info(f"Saved new confusion matrix to {matrix_output_path}")

    # 6. Compare with pre-Bible matrix
    pre_bible_matrix_path = Path(
        "data/arpabet_alignment/matrices/charliemcvicker_length-only-20260704-155307-asr-cherokee-colon_76e62140955f4738abdab345ea34068b02d8d2a2_confusion_matrix.json"
    )
    if pre_bible_matrix_path.exists():
        logger.info(
            "Generating comparative visualization against pre-Bible baseline..."
        )
        pre_bible_matrix = AcousticConfusionMatrix.load(pre_bible_matrix_path)
        output_png = Path(
            "data/arpabet_alignment/visualizations/arpabet_comparison_prebible_vs_ckpt8000.png"
        )
        generate_comparison_visualizations(pre_bible_matrix, trained_matrix, output_png)

    # Print summary comparison
    print("\n" + "=" * 80)
    print("ARPABET-TO-CHEROKEE ACOUSTIC MAPPING COMPARISON")
    print("=" * 80)
    print(
        f"{'ARPAbet':<10} | {'Pre-Bible Target':<18} | {'Checkpoint-8000 Target':<22} | {'Shift':<10}"
    )
    print("-" * 80)

    for p in STANDARD_ARPABET_PHONEMES:
        old_target, old_p = (
            pre_bible_matrix.best_target_for(p)
            if pre_bible_matrix_path.exists()
            else ("-", 0.0)
        )
        new_target, new_p = trained_matrix.best_target_for(p)
        shift_note = (
            "IDENTICAL"
            if old_target == new_target
            else f"CHANGED: {old_target}->{new_target}"
        )
        print(
            f"{p:<10} | {old_target} ({old_p*100:.1f}%) {'':<6} | {new_target} ({new_p*100:.1f}%) {'':<8} | {shift_note}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
