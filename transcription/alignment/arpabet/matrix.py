# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.matrix module.

Articulatory phonetic seed initialization, Numba-accelerated Wagner-Fischer
dynamic programming alignment, and iterative Expectation-Maximization (EM)
statistical matrix estimation for ARPAbet-to-Cherokee phonetic mapping.

Follows Types and Maps architectural principles:
- Deterministic articulatory distance mappings based on place and manner of articulation.
- Numba JIT-compiled dynamic programming kernel for microsecond-scale word alignment.
- Confidence-weighted iterative EM frequency accumulation converging P(Cherokee | ARPAbet),
  epenthetic insertion costs, and coda deletion costs in 3-5 cycles.
- Transition probability pruning (< 5%) with normalized negative log-cost serialization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numba
import numpy as np

from transcription.alignment.arpabet.dataset import load_words_manifest
from transcription.alignment.arpabet.inference import sanitize_model_id
from transcription.alignment.arpabet.types import (
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    AcousticConfusionMatrix,
    AlignedTokenPair,
    ArpabetToken,
    CherokeeToken,
    InferenceCacheManifest,
    TracebackAlignerProtocol,
    TracebackAlignmentResult,
    WordInferenceCacheEntry,
    WordManifestEntry,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Articulatory Phonetic Distance Tables
# ============================================================================

# Explicit place and manner distances between ARPAbet and Cherokee phonemes.
# Close matches: 0.2 - 0.5. Distant same-category matches: 1.0 - 1.8. Cross-category: 2.4.
ARTICULATORY_FEATURE_DISTANCES: Dict[str, Dict[str, float]] = {
    # ------------------------------------------------------------------------
    # ARPAbet Vowels & Diphthongs (15 standard + multi-token targets)
    # ------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------
    # ARPAbet Stops (Voiced & Voiceless) (6)
    # ------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------
    # ARPAbet Sibilants, Fricatives & Affricates (6)
    # ------------------------------------------------------------------------
    "CH": {"tsh": 0.2, "ts": 0.3, "s": 0.5, "hs": 0.5, "th": 0.7},
    "JH": {"ts": 0.2, "tsh": 0.4, "s": 0.5, "hs": 0.5, "t": 0.7},
    "S": {"s": 0.2, "hs": 0.3, "ts": 0.5, "tsh": 0.6, "th": 0.8},
    "Z": {"s": 0.2, "hs": 0.3, "ts": 0.4, "tsh": 0.5},
    "SH": {"s": 0.2, "hs": 0.3, "ts": 0.4, "tsh": 0.4},
    "ZH": {"s": 0.3, "hs": 0.4, "ts": 0.3, "tsh": 0.4},
    # ------------------------------------------------------------------------
    # Dental & Labiodental Fricatives (4)
    # ------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------
    # Nasals (3)
    # ------------------------------------------------------------------------
    "M": {"m": 0.2, "n": 0.5, "nh": 0.6, "w": 0.7},
    "N": {"n": 0.2, "nh": 0.4, "m": 0.5, "l": 0.7},
    "NG": {"nv": 0.2, "nk": 0.3, "n": 0.3, "nh": 0.4, "k": 0.5, "kh": 0.6, "m": 0.6},
    # ------------------------------------------------------------------------
    # Liquids (2)
    # ------------------------------------------------------------------------
    "L": {"l": 0.2, "lh": 0.3, "tl": 0.4, "tlh": 0.5, "w": 0.7},
    "R": {"l": 0.3, "lh": 0.4, "w": 0.4, "wh": 0.5, "tl": 0.6, "tlh": 0.7},
    # ------------------------------------------------------------------------
    # Glides & Laryngeals (3)
    # ------------------------------------------------------------------------
    "W": {"w": 0.2, "wh": 0.3, "kw": 0.4, "kwh": 0.5, "m": 0.7},
    "Y": {"y": 0.2, "yh": 0.3, "ts": 0.6, "i": 0.8},
    "HH": {"h": 0.2, "hs": 0.3, "'": 0.4, "wh": 0.5, "yh": 0.5},
    # ------------------------------------------------------------------------
    # Multi-gram 2-Phone Clusters
    # ------------------------------------------------------------------------
    "S T": {"hst": 0.2, "st": 0.3, "t": 0.5, "s": 0.5},
    "S K": {"hsk": 0.2, "hskw": 0.25, "sk": 0.3, "kw": 0.4},
    "S P": {"hskw": 0.25, "hsp": 0.3, "kw": 0.4, "wh": 0.4},
    "T SH": {"tsh": 0.2, "ts": 0.3},
    "D ZH": {"ts": 0.2, "tsh": 0.4},
    "N G": {"nv": 0.2, "nk": 0.3, "n": 0.3},
}


def get_articulatory_distance(
    arpabet_phone: str,
    cherokee_phone: str,
) -> float:
    """
    Computes articulatory distance between an ARPAbet phoneme (or multi-gram)
    and a Cherokee token sequence.

    Returns:
        float distance value in [0.2, 2.4].
    """
    a = arpabet_phone.strip().upper()
    c = cherokee_phone.strip().lower()

    # Check explicit mapped distances
    if a in ARTICULATORY_FEATURE_DISTANCES:
        if c in ARTICULATORY_FEATURE_DISTANCES[a]:
            return ARTICULATORY_FEATURE_DISTANCES[a][c]

    tokens_a = a.split()
    if len(tokens_a) == 2 and len(c) >= 2:
        c1, c2 = c[:1], c[1:]
        d1 = get_articulatory_distance(tokens_a[0], c1)
        d2 = get_articulatory_distance(tokens_a[1], c2)
        return (d1 + d2) / 2.0

    is_a_vowel = a in STANDARD_ARPABET_VOWELS
    is_c_vowel = c in CANONICAL_CHEROKEE_VOWELS

    # Cross-category penalty
    if is_a_vowel != is_c_vowel:
        return 2.4

    # Unmapped same-category defaults
    return 1.5 if is_a_vowel else 1.8


# ============================================================================
# 2. Articulatory Seed Matrix Builder
# ============================================================================


def build_articulatory_seed_matrix(
    arpabet_phonemes: Optional[Sequence[str]] = None,
    cherokee_phonemes: Optional[Sequence[str]] = None,
    model_id: str = "articulatory_seed",
    temperature: float = 0.5,
) -> AcousticConfusionMatrix:
    """
    Builds an initial AcousticConfusionMatrix using articulatory phonetic feature similarity.

    Maps place and manner of articulation into normalized conditional probabilities
    P(Cherokee | ARPAbet) including common multi-gram patterns (diphthongs and clusters).

    Args:
        arpabet_phonemes: Optional sequence of ARPAbet phoneme symbols (defaults to STANDARD_ARPABET_PHONEMES + seeds).
        cherokee_phonemes: Optional sequence of Cherokee phoneme symbols (defaults to CANONICAL_CHEROKEE_PHONEMES).
        model_id: Model identifier string.
        temperature: Softmax scaling temperature for distance-to-probability mapping (default 0.5).

    Returns:
        Initialized AcousticConfusionMatrix instance with precomputed log costs.
    """
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

    all_seed_keys = list(arp_vocab)
    for k in ARTICULATORY_FEATURE_DISTANCES.keys():
        if k not in all_seed_keys:
            all_seed_keys.append(k)

    # 1. Compute substitution conditional probabilities P(Cherokee | ARPAbet)
    probabilities: Dict[str, Dict[str, float]] = {}
    for a in arp_vocab:
        weights: Dict[str, float] = {}
        for c in chr_vocab:
            dist = get_articulatory_distance(a, c)
            weights[c] = math.exp(-dist / max(temperature, 1e-4))

        sum_w = sum(weights.values())
        if sum_w > 0:
            probabilities[a] = {c: w / sum_w for c, w in weights.items()}

    # Initialize any extra multi-gram distance seeds
    for k, dist_map in ARTICULATORY_FEATURE_DISTANCES.items():
        if k not in probabilities:
            weights = {
                c: math.exp(-dist / max(temperature, 1e-4))
                for c, dist in dist_map.items()
            }
            sum_w = sum(weights.values())
            if sum_w > 0:
                probabilities[k] = {c: w / sum_w for c, w in weights.items()}

    # 2. Compute epenthetic insertion probabilities P(Cherokee | <eps>)
    # Cherokee has strong CV phonotactics; epenthetic vowels (i, a, u, v, e, o)
    # and laryngeals (h, hs, ') are common insertion sites.
    ins_weights: Dict[str, float] = {}
    for c in chr_vocab:
        if c == "i":
            ins_weights[c] = 4.0
        elif c == "a":
            ins_weights[c] = 3.5
        elif c in ("u", "v"):
            ins_weights[c] = 2.5
        elif c in ("e", "o"):
            ins_weights[c] = 1.5
        elif c in ("h", "hs", "'"):
            ins_weights[c] = 1.0
        else:
            ins_weights[c] = 0.1

    sum_ins = sum(ins_weights.values())
    insertion_probabilities: Dict[str, float] = {
        c: w / sum_ins for c, w in ins_weights.items()
    }

    # 3. Compute coda deletion probabilities P(<eps> | ARPAbet)
    deletion_probabilities: Dict[str, float] = {}
    coda_frequent = {
        "T",
        "D",
        "P",
        "K",
        "B",
        "G",
        "S",
        "Z",
        "SH",
        "ZH",
        "L",
        "R",
        "N",
        "M",
    }
    weak_vowels = {"AH", "ER", "IH"}

    for a in arp_vocab:
        if a in coda_frequent:
            deletion_probabilities[a] = 0.25
        elif a in STANDARD_ARPABET_CONSONANTS:
            deletion_probabilities[a] = 0.18
        elif a in weak_vowels:
            deletion_probabilities[a] = 0.10
        else:
            deletion_probabilities[a] = 0.05

    return AcousticConfusionMatrix.create(
        model_id=model_id,
        probabilities=probabilities,
        insertion_probabilities=insertion_probabilities,
        deletion_probabilities=deletion_probabilities,
        arpabet_vocab=arp_vocab,
        cherokee_vocab=chr_vocab,
        prune_threshold=0.05,
        iteration=0,
        metadata={"source": "articulatory_feature_seed", "temperature": temperature},
    )


# ============================================================================
# 3. Numba-Accelerated Wagner-Fischer Kernel
# ============================================================================


@numba.njit
def _wagner_fischer_kernel(
    sub_costs: np.ndarray,
    ins_costs: np.ndarray,
    del_costs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba JIT-compiled dynamic programming kernel for Wagner-Fischer traceback alignment.

    Args:
        sub_costs: 2D array of substitution costs of shape (N, M).
        ins_costs: 1D array of insertion costs of shape (M,).
        del_costs: 1D array of deletion costs of shape (N,).

    Returns:
        Tuple of:
            - dp: 2D float64 array of shape (N + 1, M + 1) with minimum cumulative costs.
            - backpointers: 2D int8 array of shape (N + 1, M + 1) where:
                0 = start/boundary
                1 = substitution (diagonal: i-1, j-1)
                2 = deletion (up: i-1, j)
                3 = insertion (left: i, j-1)
    """
    N, M = sub_costs.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float64)
    backpointers = np.zeros((N + 1, M + 1), dtype=np.int8)

    for i in range(1, N + 1):
        dp[i, 0] = dp[i - 1, 0] + del_costs[i - 1]
        backpointers[i, 0] = 2  # del

    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + ins_costs[j - 1]
        backpointers[0, j] = 3  # ins

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost_sub = dp[i - 1, j - 1] + sub_costs[i - 1, j - 1]
            cost_del = dp[i - 1, j] + del_costs[i - 1]
            cost_ins = dp[i, j - 1] + ins_costs[j - 1]

            best_cost = cost_sub
            bp = 1  # sub
            if cost_del < best_cost:
                best_cost = cost_del
                bp = 2  # del
            if cost_ins < best_cost:
                best_cost = cost_ins
                bp = 3  # ins

            dp[i, j] = best_cost
            backpointers[i, j] = bp

    return dp, backpointers


# ============================================================================
# 4. Confidence-Weighted DP Traceback Aligner
# ============================================================================


def align_word_pair(
    arpabet_tokens: Sequence[Union[str, ArpabetToken]],
    cherokee_tokens: Sequence[Union[str, CherokeeToken]],
    matrix: AcousticConfusionMatrix,
    token_confidences: Optional[Sequence[float]] = None,
) -> TracebackAlignmentResult:
    """
    Performs dynamic programming alignment between an ARPAbet phoneme sequence
    and emitted Cherokee tokens using negative log-costs from AcousticConfusionMatrix.
    Supports 1-to-1, 1-to-2, 2-to-1, and 2-to-2 multi-gram transitions.

    Confidence weighting:
    - Attaches token confidence to each AlignedTokenPair for subsequent weighted EM accumulation.
    - Accurately recovers substitutions, multi-token expansions/fusions, epenthetic insertions, and deletions.

    Args:
        arpabet_tokens: Sequence of input ARPAbet phonemes (strings or ArpabetToken).
        cherokee_tokens: Sequence of emitted Cherokee phonemes (strings or CherokeeToken).
        matrix: Calibrated or seeded AcousticConfusionMatrix.
        token_confidences: Optional per-Cherokee-token confidence scores in [0.0, 1.0].

    Returns:
        TracebackAlignmentResult with aligned token pairs, total cost, and normalized cost.
    """
    # Normalize input tokens
    arp_objs: List[ArpabetToken] = [
        t if isinstance(t, ArpabetToken) else ArpabetToken(t)
        for t in arpabet_tokens
        if (t.phone if isinstance(t, ArpabetToken) else t)
        not in ("", "<eps>", "<EPS>", "eps", "EPS")
    ]
    chr_objs: List[CherokeeToken] = [
        t if isinstance(t, CherokeeToken) else CherokeeToken(t)
        for t in cherokee_tokens
        if (t.phone if isinstance(t, CherokeeToken) else t)
        not in ("", "<eps>", "<EPS>", "eps", "EPS")
    ]

    N = len(arp_objs)
    M = len(chr_objs)

    # Normalize confidences for Cherokee emissions
    confs: List[float] = (
        list(token_confidences) if token_confidences is not None else []
    )
    if len(confs) < M:
        confs.extend([1.0] * (M - len(confs)))

    # Handle boundary conditions
    if N == 0 and M == 0:
        return TracebackAlignmentResult(
            pairs=(),
            total_cost=0.0,
            normalized_cost=0.0,
        )

    if N == 0:
        ins_pairs: List[AlignedTokenPair] = [
            AlignedTokenPair(
                arpabet=None,
                cherokee=chr_objs[j],
                arpabet_tokens=(),
                cherokee_tokens=(chr_objs[j],),
                cost=matrix.get_insertion_cost(chr_objs[j]),
                confidence=confs[j],
            )
            for j in range(M)
        ]
        total_cost = sum(p.cost for p in ins_pairs)
        return TracebackAlignmentResult(
            pairs=tuple(ins_pairs),
            total_cost=total_cost,
            normalized_cost=total_cost / max(1, len(ins_pairs)),
        )

    if M == 0:
        del_pairs: List[AlignedTokenPair] = [
            AlignedTokenPair(
                arpabet=arp_objs[i],
                cherokee=None,
                arpabet_tokens=(arp_objs[i],),
                cherokee_tokens=(),
                cost=matrix.get_deletion_cost(arp_objs[i]),
                confidence=1.0,
            )
            for i in range(N)
        ]
        total_cost = sum(p.cost for p in del_pairs)
        return TracebackAlignmentResult(
            pairs=tuple(del_pairs),
            total_cost=total_cost,
            normalized_cost=total_cost / max(1, len(del_pairs)),
        )

    # Dynamic programming grid supporting (1->1), (1->0), (0->1), (1->2), (2->1), (2->2)
    dp = np.full((N + 1, M + 1), 1e9, dtype=np.float64)
    backpointers_i = np.zeros((N + 1, M + 1), dtype=np.int8)
    backpointers_j = np.zeros((N + 1, M + 1), dtype=np.int8)

    dp[0, 0] = 0.0

    for i in range(1, N + 1):
        dp[i, 0] = dp[i - 1, 0] + matrix.get_deletion_cost(arp_objs[i - 1].phone)
        backpointers_i[i, 0] = 1
        backpointers_j[i, 0] = 0

    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + matrix.get_insertion_cost(chr_objs[j - 1].phone)
        backpointers_i[0, j] = 0
        backpointers_j[0, j] = 1

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # 1. Substitution (1 -> 1)
            best_cost = dp[i - 1, j - 1] + matrix.get_substitution_cost(
                arp_objs[i - 1].phone, chr_objs[j - 1].phone
            )
            best_di = 1
            best_dj = 1

            # 2. Deletion (1 -> 0)
            cost_del = dp[i - 1, j] + matrix.get_deletion_cost(arp_objs[i - 1].phone)
            if cost_del < best_cost:
                best_cost = cost_del
                best_di = 1
                best_dj = 0

            # 3. Insertion (0 -> 1)
            cost_ins = dp[i, j - 1] + matrix.get_insertion_cost(chr_objs[j - 1].phone)
            if cost_ins < best_cost:
                best_cost = cost_ins
                best_di = 0
                best_dj = 1

            # 4. Expansion (1 -> 2)
            if j >= 2:
                c2_key = f"{chr_objs[j - 2].phone}{chr_objs[j - 1].phone}"
                cost_exp = dp[i - 1, j - 2] + matrix.get_substitution_cost(
                    arp_objs[i - 1].phone, c2_key
                )
                if cost_exp < best_cost:
                    best_cost = cost_exp
                    best_di = 1
                    best_dj = 2

            # 5. Fusion (2 -> 1)
            if i >= 2:
                a2_key = f"{arp_objs[i - 2].phone} {arp_objs[i - 1].phone}"
                cost_fus = dp[i - 2, j - 1] + matrix.get_substitution_cost(
                    a2_key, chr_objs[j - 1].phone
                )
                if cost_fus < best_cost:
                    best_cost = cost_fus
                    best_di = 2
                    best_dj = 1

            # 6. Joint 2-gram (2 -> 2)
            if i >= 2 and j >= 2:
                a2_key = f"{arp_objs[i - 2].phone} {arp_objs[i - 1].phone}"
                c2_key = f"{chr_objs[j - 2].phone}{chr_objs[j - 1].phone}"
                cost_joint = dp[i - 2, j - 2] + matrix.get_substitution_cost(
                    a2_key, c2_key
                )
                if cost_joint < best_cost:
                    best_cost = cost_joint
                    best_di = 2
                    best_dj = 2

            dp[i, j] = best_cost
            backpointers_i[i, j] = best_di
            backpointers_j[i, j] = best_dj

    # Traceback from (N, M) to (0, 0)
    pairs: List[AlignedTokenPair] = []
    i, j = N, M
    while i > 0 or j > 0:
        di = int(backpointers_i[i, j])
        dj = int(backpointers_j[i, j])
        if di == 0 and dj == 0:
            break
        step_cost = float(dp[i, j] - dp[i - di, j - dj])
        arp_slice = tuple(arp_objs[i - di : i]) if di > 0 else ()
        chr_slice = tuple(chr_objs[j - dj : j]) if dj > 0 else ()
        conf = float(np.mean([confs[k] for k in range(j - dj, j)])) if dj > 0 else 1.0
        pairs.append(
            AlignedTokenPair(
                arpabet=(
                    arp_slice[0]
                    if len(arp_slice) == 1
                    else (
                        ArpabetToken(" ".join(t.phone for t in arp_slice))
                        if arp_slice
                        else None
                    )
                ),
                cherokee=(
                    chr_slice[0]
                    if len(chr_slice) == 1
                    else (
                        CherokeeToken("".join(t.phone for t in chr_slice))
                        if chr_slice
                        else None
                    )
                ),
                arpabet_tokens=arp_slice,
                cherokee_tokens=chr_slice,
                cost=step_cost,
                confidence=conf,
            )
        )
        i -= di
        j -= dj

    pairs.reverse()
    total_cost = float(dp[N, M])
    normalized_cost = total_cost / max(1, len(pairs))

    return TracebackAlignmentResult(
        pairs=tuple(pairs),
        total_cost=total_cost,
        normalized_cost=normalized_cost,
    )


class WagnerFischerAligner:
    """
    Pure functional alignment service implementing TracebackAlignerProtocol.
    """

    def align(
        self,
        arpabet_tokens: Sequence[Union[str, ArpabetToken]],
        cherokee_tokens: Sequence[Union[str, CherokeeToken]],
        matrix: AcousticConfusionMatrix,
        token_confidences: Optional[Sequence[float]] = None,
    ) -> TracebackAlignmentResult:
        """Aligns ARPAbet and Cherokee token sequences via Wagner-Fischer DP."""
        return align_word_pair(
            arpabet_tokens=arpabet_tokens,
            cherokee_tokens=cherokee_tokens,
            matrix=matrix,
            token_confidences=token_confidences,
        )


# ============================================================================
# 5. Iterative Expectation-Maximization (EM) Matrix Estimator
# ============================================================================


def train_acoustic_confusion_matrix(
    manifest_entries: Union[
        Sequence[WordManifestEntry], Path, str, List[Dict[str, Any]]
    ],
    emissions_manifest: Union[InferenceCacheManifest, Path, str, List[Dict[str, Any]]],
    num_iterations: int = 4,
    prune_threshold: float = 0.05,
    seed_matrix: Optional[AcousticConfusionMatrix] = None,
    model_id: Optional[str] = None,
    alpha_prior: float = 0.05,
) -> AcousticConfusionMatrix:
    """
    Trains an AcousticConfusionMatrix using iterative Expectation-Maximization (EM).

    Algorithm:
    1. Initialize from articulatory feature seed matrix (seed_matrix).
    2. E-step: Aligns all dataset word pairs using current DP costs.
       Accumulates confidence-weighted transition counts Count(Cherokee | ARPAbet),
       including multi-gram (1-to-2, 2-to-1, 2-to-2) transitions,
       epenthetic insertion counts Count(Cherokee | <eps>), and coda deletion counts Count(<eps> | ARPAbet).
    3. M-step: Normalizes frequency counts into conditional probability distributions
       P(Cherokee | ARPAbet), P(Cherokee | <eps>), and P(<eps> | ARPAbet).
    4. Repeats for num_iterations cycles.
    5. Pruning: Prunes transitions below prune_threshold (< 5%), re-normalizes surviving
       distributions, and converts to negative log-costs.

    Args:
        manifest_entries: WordManifestEntry sequence, list of dicts, or path to words_manifest.json.
        emissions_manifest: InferenceCacheManifest, list of dicts, or path to emissions cache JSON.
        num_iterations: Number of EM cycles to run (default 4, typical 3-5).
        prune_threshold: Minimum transition probability threshold to retain (default 0.05).
        seed_matrix: Optional pre-initialized seed matrix. If None, builds articulatory seed.
        model_id: Model identifier for metadata and serialization.
        alpha_prior: Dirichlet prior smoothing weight to maintain stability for rare tokens (default 0.05).

    Returns:
        Finalized, pruned, and normalized AcousticConfusionMatrix.
    """
    # 1. Ingest Word Manifest
    if isinstance(manifest_entries, (str, Path)):
        word_entries = load_words_manifest(manifest_entries)
    elif isinstance(manifest_entries, Sequence):
        word_entries = [
            e if isinstance(e, WordManifestEntry) else WordManifestEntry.from_dict(e)
            for e in manifest_entries
        ]
    else:
        raise TypeError(f"Unsupported manifest_entries type: {type(manifest_entries)}")

    manifest_by_id: Dict[str, WordManifestEntry] = {e.clip_id: e for e in word_entries}

    # 2. Ingest Emissions Cache Manifest
    if isinstance(emissions_manifest, (str, Path)):
        cache = InferenceCacheManifest.load(emissions_manifest)
    elif isinstance(emissions_manifest, InferenceCacheManifest):
        cache = emissions_manifest
    elif isinstance(emissions_manifest, (list, tuple)):
        entries = [
            (
                e
                if isinstance(e, WordInferenceCacheEntry)
                else WordInferenceCacheEntry.from_dict(e)
            )
            for e in emissions_manifest
        ]
        cache = InferenceCacheManifest(
            model_id=model_id or "unknown_model",
            created_at=datetime.now(timezone.utc).isoformat(),
            entries=tuple(entries),
        )
    else:
        raise TypeError(
            f"Unsupported emissions_manifest type: {type(emissions_manifest)}"
        )

    emissions_by_id: Dict[str, WordInferenceCacheEntry] = cache.by_clip_id

    effective_model_id = model_id or cache.model_id or "charliemcvicker_cherokee_asr"

    # 3. Match paired data
    paired_data: List[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[float, ...]]] = []
    for clip_id, man_entry in manifest_by_id.items():
        if clip_id in emissions_by_id:
            em_entry = emissions_by_id[clip_id]
            arp_phones = tuple(
                tok.phone for tok in man_entry.arpabet if not tok.is_epsilon
            )
            chr_phones = tuple(
                tok.phone for tok in em_entry.greedy_tokens if not tok.is_epsilon
            )
            confs = tuple(float(c) for c in em_entry.token_confidences)
            if arp_phones or chr_phones:
                paired_data.append((arp_phones, chr_phones, confs))

    logger.info(
        f"Initialized EM training with {len(paired_data)} word pairs for model '{effective_model_id}'."
    )

    # 4. Initialize current matrix from articulatory seed
    current_matrix: AcousticConfusionMatrix = (
        seed_matrix
        if seed_matrix is not None
        else build_articulatory_seed_matrix(model_id=effective_model_id)
    )

    arp_vocab: Tuple[str, ...] = current_matrix.arpabet_vocab
    chr_vocab: Tuple[str, ...] = current_matrix.cherokee_vocab

    last_delta: float = 0.0
    mean_cost: float = 0.0

    # 5. Run EM iterations
    for iteration in range(1, num_iterations + 1):
        # --------------------------------------------------------------------
        # E-Step: Accumulate alignment counts
        # --------------------------------------------------------------------
        sub_counts: Dict[str, Dict[str, float]] = {}
        ins_counts: Dict[str, float] = {c: 0.0 for c in chr_vocab}
        del_counts: Dict[str, float] = {a: 0.0 for a in arp_vocab}

        total_cost_sum = 0.0
        total_alignments = len(paired_data)

        for arp_seq, chr_seq, confs in paired_data:
            align_res = align_word_pair(
                arp_seq, chr_seq, current_matrix, token_confidences=confs
            )
            total_cost_sum += align_res.total_cost

            for pair in align_res.pairs:
                if pair.is_substitution:
                    a_ph = pair.arpabet_key
                    c_ph = pair.cherokee_key
                    w = pair.confidence
                    if a_ph not in sub_counts:
                        sub_counts[a_ph] = {}
                    sub_counts[a_ph][c_ph] = sub_counts[a_ph].get(c_ph, 0.0) + w
                elif pair.is_insertion:
                    c_ph = pair.cherokee_key
                    w = pair.confidence
                    ins_counts[c_ph] = ins_counts.get(c_ph, 0.0) + w
                elif pair.is_deletion:
                    a_ph = pair.arpabet_key
                    del_counts[a_ph] = del_counts.get(a_ph, 0.0) + 1.0

        mean_cost = total_cost_sum / max(1, total_alignments)

        # --------------------------------------------------------------------
        # M-Step: Re-estimate conditional probabilities
        # --------------------------------------------------------------------
        new_probabilities: Dict[str, Dict[str, float]] = {}
        max_delta = 0.0
        all_arp_keys = (
            set(arp_vocab)
            | set(sub_counts.keys())
            | set(current_matrix.probabilities.keys())
        )

        for a in sorted(all_arp_keys):
            prior_map = current_matrix.probabilities.get(a, {})
            obs_map = sub_counts.get(a, {})
            all_targets = set(obs_map.keys()) | set(prior_map.keys())
            if not all_targets:
                continue

            effective_counts: Dict[str, float] = {
                c: obs_map.get(c, 0.0) + alpha_prior * prior_map.get(c, 0.01)
                for c in all_targets
            }
            total_a = sum(effective_counts.values())
            new_probabilities[a] = {}
            for c in all_targets:
                p_new = effective_counts[c] / max(total_a, 1e-12)
                new_probabilities[a][c] = p_new
                old_p = prior_map.get(c, 0.0)
                max_delta = max(max_delta, abs(p_new - old_p))

        # Re-estimate insertion probabilities P(c | <eps>)
        new_ins_probs: Dict[str, float] = {}
        prior_ins = current_matrix.insertion_probabilities
        all_ins_c = set(ins_counts.keys()) | set(prior_ins.keys()) | set(chr_vocab)
        eff_ins = {
            c: ins_counts.get(c, 0.0)
            + alpha_prior * prior_ins.get(c, 1.0 / max(1, len(all_ins_c)))
            for c in all_ins_c
        }
        tot_ins = sum(eff_ins.values())
        for c in all_ins_c:
            new_ins_probs[c] = eff_ins[c] / max(tot_ins, 1e-12)

        # Re-estimate deletion probabilities P(<eps> | a)
        new_del_probs: Dict[str, float] = {}
        prior_del = current_matrix.deletion_probabilities
        for a in all_arp_keys:
            tot_sub_a = sum(sub_counts.get(a, {}).values())
            tot_del_a = del_counts.get(a, 0.0)
            tot_a = tot_sub_a + tot_del_a
            if tot_a > 0:
                new_del_probs[a] = (tot_del_a + alpha_prior * prior_del.get(a, 0.1)) / (
                    tot_a + alpha_prior
                )
            else:
                new_del_probs[a] = prior_del.get(a, 0.1)

        last_delta = max_delta
        logger.info(
            f"EM Iteration {iteration}/{num_iterations}: delta={max_delta:.5f}, mean_cost={mean_cost:.3f}"
        )

        # Update matrix for next iteration
        current_matrix = AcousticConfusionMatrix.create(
            model_id=effective_model_id,
            probabilities=new_probabilities,
            insertion_probabilities=new_ins_probs,
            deletion_probabilities=new_del_probs,
            arpabet_vocab=tuple(sorted(new_probabilities.keys())),
            cherokee_vocab=chr_vocab,
            prune_threshold=prune_threshold,
            iteration=iteration,
            metadata={
                "iteration": iteration,
                "delta": max_delta,
                "mean_cost": mean_cost,
            },
        )

    # ------------------------------------------------------------------------
    # Pruning & Normalization Step
    # ------------------------------------------------------------------------
    pruned_probabilities: Dict[str, Dict[str, float]] = {}
    for a in sorted(current_matrix.probabilities.keys()):
        raw_map = current_matrix.probabilities.get(a, {})
        # Filter transitions above threshold
        filtered = {c: p for c, p in raw_map.items() if p >= prune_threshold}
        if not filtered and raw_map:
            # Fallback to argmax if everything was pruned
            best_c, best_p = max(raw_map.items(), key=lambda kv: kv[1])
            filtered = {best_c: best_p}

        if filtered:
            sum_f = sum(filtered.values())
            pruned_probabilities[a] = {c: p / sum_f for c, p in filtered.items()}

    # Prune insertion probabilities
    raw_ins = current_matrix.insertion_probabilities
    filtered_ins = {c: p for c, p in raw_ins.items() if p >= prune_threshold}
    if not filtered_ins and raw_ins:
        best_c, best_p = max(raw_ins.items(), key=lambda kv: kv[1])
        filtered_ins = {best_c: best_p}
    sum_ins_f = sum(filtered_ins.values())
    pruned_insertion = {c: p / sum_ins_f for c, p in filtered_ins.items()}

    # Clip deletion probabilities to valid probability range
    pruned_deletion = {
        a: max(0.01, min(0.99, p))
        for a, p in current_matrix.deletion_probabilities.items()
    }

    final_matrix = AcousticConfusionMatrix.create(
        model_id=effective_model_id,
        probabilities=pruned_probabilities,
        insertion_probabilities=pruned_insertion,
        deletion_probabilities=pruned_deletion,
        arpabet_vocab=tuple(sorted(pruned_probabilities.keys())),
        cherokee_vocab=chr_vocab,
        prune_threshold=prune_threshold,
        iteration=num_iterations,
        metadata={
            "trained_on_pairs": len(paired_data),
            "num_iterations": num_iterations,
            "prune_threshold": prune_threshold,
            "final_delta": last_delta,
            "mean_cost": mean_cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return final_matrix


# ============================================================================
# 6. CLI Runner
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and serialize ARPAbet-to-Cherokee empirical acoustic confusion matrix."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/arpabet_alignment/words_manifest.json",
        help="Path to LibriSpeech words manifest JSON.",
    )
    parser.add_argument(
        "--emissions-path",
        type=str,
        default="data/arpabet_alignment/cache/charliemcvicker_length-only-20260704-155307-asr-cherokee-colon_76e62140955f4738abdab345ea34068b02d8d2a2_emissions.json",
        help="Path to cached ASR emissions manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/arpabet_alignment/matrices",
        help="Directory to save the trained confusion matrix.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Explicit output path for the serialized confusion matrix JSON.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=4,
        help="Number of Expectation-Maximization iterations (default: 4).",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.05,
        help="Probability threshold below which transitions are pruned (default: 0.05).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    manifest_p = Path(args.manifest_path)
    emissions_p = Path(args.emissions_path)

    if not manifest_p.exists():
        raise FileNotFoundError(f"Words manifest not found: {manifest_p}")
    if not emissions_p.exists():
        raise FileNotFoundError(f"Emissions manifest not found: {emissions_p}")

    logger.info(f"Loading emissions from {emissions_p}...")
    cache = InferenceCacheManifest.load(emissions_p)
    sanitized_id = sanitize_model_id(cache.model_id)

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sanitized_id}_confusion_matrix.json"

    logger.info(f"Training acoustic confusion matrix for '{cache.model_id}'...")
    matrix = train_acoustic_confusion_matrix(
        manifest_entries=manifest_p,
        emissions_manifest=cache,
        num_iterations=args.num_iterations,
        prune_threshold=args.prune_threshold,
        model_id=cache.model_id,
    )

    logger.info(f"Saving confusion matrix to {out_path}...")
    matrix.save(out_path)
    logger.info(
        f"Done! Trained matrix serialized successfully ({len(matrix.probabilities)} ARPAbet phonemes)."
    )


if __name__ == "__main__":
    main()
