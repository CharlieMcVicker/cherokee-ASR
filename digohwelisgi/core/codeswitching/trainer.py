# -*- coding: utf-8 -*-
"""
digohwelisgi.core.codeswitching.trainer

Language-agnostic Expectation-Maximization (EM) trainer for AcousticConfusionMatrix,
dynamic programming traceback alignment, duration-sorted bucketing, and CTC prefix beam search.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numba
import numpy as np

from digohwelisgi.core.codeswitching.matrix import AcousticConfusionMatrix
from digohwelisgi.core.codeswitching.types import (
    EPSILON_TOKEN,
    ARPAbetPhone,
    AlignmentPath,
    BeamHypothesis,
    CrossEntropyLoss,
    TracebackResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sanitize_model_id(model_id: str) -> str:
    """Sanitizes a model identifier for safe filesystem path naming."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", model_id)
    cleaned = cleaned.strip("._")
    return cleaned or "default_model"


def bucket_by_duration(
    manifest_entries: Sequence[T],
    batch_size: int = 32,
    duration_key: Optional[Callable[[T], float]] = None,
) -> List[List[T]]:
    """
    Sorts entries by duration and partitions them into batches of at most batch_size.
    Minimizes wasted padding frames during batched ASR model forward passes.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not manifest_entries:
        return []

    def _get_duration(item: T) -> float:
        if duration_key is not None:
            return duration_key(item)
        if hasattr(item, "duration"):
            return float(getattr(item, "duration"))
        if isinstance(item, dict) and "duration" in item:
            return float(item["duration"])
        return 0.0

    sorted_entries = sorted(manifest_entries, key=_get_duration)
    return [
        list(sorted_entries[i : i + batch_size])
        for i in range(0, len(sorted_entries), batch_size)
    ]


def _log_add(a: float, b: float) -> float:
    """Computes log(exp(a) + exp(b)) in a numerically stable manner."""
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    logits: np.ndarray,
    vocab: Sequence[str],
    pad_id: int = 0,
    beam_width: int = 10,
    top_k: int = 3,
    prune_threshold: float = 1e-4,
) -> List[BeamHypothesis]:
    """
    Language-agnostic CTC prefix beam search decoding for acoustic logits.
    """
    if logits.ndim == 3:
        logits = logits.squeeze(0)

    max_val = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_val)
    log_probs = logits - max_val - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

    T, V = log_probs.shape
    if T == 0:
        return [
            BeamHypothesis(rank=1, text="", score=0.0, tokens=(), token_confidences=())
        ]

    log_prune_threshold = math.log(prune_threshold)

    # State: prefix -> (p_blank, p_non_blank, char_confidences)
    current_beam: Dict[Tuple[int, ...], Tuple[float, float, Tuple[float, ...]]] = {
        (): (0.0, -float("inf"), ())
    }

    for t in range(T):
        next_beam: Dict[Tuple[int, ...], Tuple[float, float, Tuple[float, ...]]] = {}
        l_t = log_probs[t]

        c_candidates: List[int] = [pad_id]
        for c in range(V):
            if c != pad_id and l_t[c] >= log_prune_threshold:
                c_candidates.append(c)

        for p, (p_b, p_nb, confs) in current_beam.items():
            p_total = _log_add(p_b, p_nb)

            # Blank transition
            new_p_b = p_total + float(l_t[pad_id])
            curr_b, curr_nb, curr_confs = next_beam.get(
                p, (-float("inf"), -float("inf"), confs)
            )
            next_beam[p] = (_log_add(curr_b, new_p_b), curr_nb, curr_confs)

            # Non-blank transitions
            for c in c_candidates:
                if c == pad_id:
                    continue
                p_c = float(l_t[c])
                prob_c = math.exp(p_c)

                if len(p) > 0 and c == p[-1]:
                    new_p = p + (c,)
                    new_nb_from_b = p_b + p_c
                    new_confs = confs + (prob_c,)
                    cb, cnb, cc = next_beam.get(
                        new_p, (-float("inf"), -float("inf"), new_confs)
                    )
                    next_beam[new_p] = (cb, _log_add(cnb, new_nb_from_b), cc)

                    new_nb_from_nb = p_nb + p_c
                    cb, cnb, cc = next_beam.get(
                        p, (-float("inf"), -float("inf"), confs)
                    )
                    upd_cc = cc[:-1] + (max(cc[-1], prob_c),) if cc else (prob_c,)
                    next_beam[p] = (cb, _log_add(cnb, new_nb_from_nb), upd_cc)
                else:
                    new_p = p + (c,)
                    new_nb = p_total + p_c
                    new_confs = confs + (prob_c,)
                    cb, cnb, cc = next_beam.get(
                        new_p, (-float("inf"), -float("inf"), new_confs)
                    )
                    next_beam[new_p] = (cb, _log_add(cnb, new_nb), cc)

        sorted_beam = sorted(
            next_beam.items(),
            key=lambda item: _log_add(item[1][0], item[1][1]),
            reverse=True,
        )
        current_beam = dict(sorted_beam[:beam_width])

    sorted_results = sorted(
        current_beam.items(),
        key=lambda item: _log_add(item[1][0], item[1][1]),
        reverse=True,
    )

    hypotheses: List[BeamHypothesis] = []
    for rank, (prefix, (pb, pnb, confs)) in enumerate(sorted_results[:top_k], start=1):
        total_score = _log_add(pb, pnb)
        tok_strs = tuple(vocab[idx] if idx < len(vocab) else "" for idx in prefix)
        text = "".join(tok_strs)
        hypotheses.append(
            BeamHypothesis(
                rank=rank,
                text=text,
                score=total_score,
                tokens=tok_strs,
                token_confidences=confs,
            )
        )

    if not hypotheses:
        hypotheses.append(
            BeamHypothesis(rank=1, text="", score=0.0, tokens=(), token_confidences=())
        )

    return hypotheses


def align_word_pair_generic(
    source_tokens: Sequence[Union[str, ARPAbetPhone]],
    target_tokens: Sequence[str],
    matrix: AcousticConfusionMatrix,
    token_confidences: Optional[Sequence[float]] = None,
) -> TracebackResult:
    """
    Generic dynamic programming alignment between source phone sequence
    and target token sequence using negative log-costs from AcousticConfusionMatrix.
    Supports 1-to-1, 1-to-2, 2-to-1, and 2-to-2 multi-gram transitions.
    """
    src_objs: List[str] = [
        t.phone if isinstance(t, ARPAbetPhone) else str(t).strip()
        for t in source_tokens
        if (t.phone if isinstance(t, ARPAbetPhone) else str(t).strip())
        not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
    ]
    tgt_objs: List[str] = [
        str(t).strip()
        for t in target_tokens
        if str(t).strip() not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
    ]

    N = len(src_objs)
    M = len(tgt_objs)

    confs: List[float] = (
        list(token_confidences) if token_confidences is not None else []
    )
    if len(confs) < M:
        confs.extend([1.0] * (M - len(confs)))

    if N == 0 and M == 0:
        return TracebackResult(paths=(), total_cost=0.0, normalized_cost=0.0)

    if N == 0:
        ins_paths: List[AlignmentPath] = [
            AlignmentPath(
                source_tokens=(),
                target_tokens=(tgt_objs[j],),
                cost=matrix.get_insertion_cost(tgt_objs[j]),
                confidence=confs[j],
            )
            for j in range(M)
        ]
        total_cost = sum(p.cost for p in ins_paths)
        return TracebackResult(
            paths=tuple(ins_paths),
            total_cost=total_cost,
            normalized_cost=total_cost / max(1, len(ins_paths)),
        )

    if M == 0:
        del_paths: List[AlignmentPath] = [
            AlignmentPath(
                source_tokens=(src_objs[i],),
                target_tokens=(),
                cost=matrix.get_deletion_cost(src_objs[i]),
                confidence=1.0,
            )
            for i in range(N)
        ]
        total_cost = sum(p.cost for p in del_paths)
        return TracebackResult(
            paths=tuple(del_paths),
            total_cost=total_cost,
            normalized_cost=total_cost / max(1, len(del_paths)),
        )

    dp = np.full((N + 1, M + 1), 1e9, dtype=np.float64)
    backpointers_i = np.zeros((N + 1, M + 1), dtype=np.int8)
    backpointers_j = np.zeros((N + 1, M + 1), dtype=np.int8)

    dp[0, 0] = 0.0

    for i in range(1, N + 1):
        dp[i, 0] = dp[i - 1, 0] + matrix.get_deletion_cost(src_objs[i - 1])
        backpointers_i[i, 0] = 1
        backpointers_j[i, 0] = 0

    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + matrix.get_insertion_cost(tgt_objs[j - 1])
        backpointers_i[0, j] = 0
        backpointers_j[0, j] = 1

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # 1. Substitution (1 -> 1)
            best_cost = dp[i - 1, j - 1] + matrix.get_substitution_cost(
                src_objs[i - 1], tgt_objs[j - 1]
            )
            best_di = 1
            best_dj = 1

            # 2. Deletion (1 -> 0)
            cost_del = dp[i - 1, j] + matrix.get_deletion_cost(src_objs[i - 1])
            if cost_del < best_cost:
                best_cost = cost_del
                best_di = 1
                best_dj = 0

            # 3. Insertion (0 -> 1)
            cost_ins = dp[i, j - 1] + matrix.get_insertion_cost(tgt_objs[j - 1])
            if cost_ins < best_cost:
                best_cost = cost_ins
                best_di = 0
                best_dj = 1

            # 4. Expansion (1 -> 2)
            if j >= 2:
                c2_key = f"{tgt_objs[j - 2]}{tgt_objs[j - 1]}"
                cost_exp = dp[i - 1, j - 2] + matrix.get_substitution_cost(
                    src_objs[i - 1], c2_key
                )
                if cost_exp < best_cost:
                    best_cost = cost_exp
                    best_di = 1
                    best_dj = 2

            # 5. Fusion (2 -> 1)
            if i >= 2:
                a2_key = f"{src_objs[i - 2]} {src_objs[i - 1]}"
                cost_fus = dp[i - 2, j - 1] + matrix.get_substitution_cost(
                    a2_key, tgt_objs[j - 1]
                )
                if cost_fus < best_cost:
                    best_cost = cost_fus
                    best_di = 2
                    best_dj = 1

            # 6. Joint 2-gram (2 -> 2)
            if i >= 2 and j >= 2:
                a2_key = f"{src_objs[i - 2]} {src_objs[i - 1]}"
                c2_key = f"{tgt_objs[j - 2]}{tgt_objs[j - 1]}"
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

    # Traceback
    paths: List[AlignmentPath] = []
    i, j = N, M
    while i > 0 or j > 0:
        di = int(backpointers_i[i, j])
        dj = int(backpointers_j[i, j])
        if di == 0 and dj == 0:
            break
        step_cost = float(dp[i, j] - dp[i - di, j - dj])
        src_slice = tuple(src_objs[i - di : i]) if di > 0 else ()
        tgt_slice = tuple(tgt_objs[j - dj : j]) if dj > 0 else ()
        conf = float(np.mean([confs[k] for k in range(j - dj, j)])) if dj > 0 else 1.0
        paths.append(
            AlignmentPath(
                source_tokens=src_slice,
                target_tokens=tgt_slice,
                cost=step_cost,
                confidence=conf,
            )
        )
        i -= di
        j -= dj

    paths.reverse()
    total_cost = float(dp[N, M])
    normalized_cost = total_cost / max(1, len(paths))

    return TracebackResult(
        paths=tuple(paths),
        total_cost=total_cost,
        normalized_cost=normalized_cost,
    )


class ConfusionMatrixTrainer:
    """
    Language-agnostic Expectation-Maximization (EM) trainer for AcousticConfusionMatrix.
    Takes aligned pairs of (source_tokens, target_tokens, confidences) and iteratively
    estimates conditional probability distributions.
    """

    def __init__(
        self,
        num_iterations: int = 4,
        prune_threshold: float = 0.05,
        alpha_prior: float = 0.05,
    ) -> None:
        self.num_iterations = num_iterations
        self.prune_threshold = prune_threshold
        self.alpha_prior = alpha_prior

    def train(
        self,
        paired_data: Sequence[
            Tuple[
                Sequence[Union[str, ARPAbetPhone]],
                Sequence[str],
                Optional[Sequence[float]],
            ]
        ],
        seed_matrix: AcousticConfusionMatrix,
        model_id: str = "trained_model",
    ) -> AcousticConfusionMatrix:
        """
        Trains AcousticConfusionMatrix using EM.
        """
        current_matrix = seed_matrix
        src_vocab = current_matrix.source_vocab
        tgt_vocab = current_matrix.target_vocab

        last_delta: float = 0.0
        mean_cost: float = 0.0

        for iteration in range(1, self.num_iterations + 1):
            sub_counts: Dict[str, Dict[str, float]] = {}
            ins_counts: Dict[str, float] = {c: 0.0 for c in tgt_vocab}
            del_counts: Dict[str, float] = {a: 0.0 for a in src_vocab}

            total_cost_sum = 0.0
            total_alignments = len(paired_data)

            for src_seq, tgt_seq, confs in paired_data:
                align_res = align_word_pair_generic(
                    src_seq, tgt_seq, current_matrix, token_confidences=confs
                )
                total_cost_sum += align_res.total_cost

                for path in align_res.paths:
                    if path.is_substitution:
                        s_key = path.source_key
                        t_key = path.target_key
                        w = path.confidence
                        if s_key not in sub_counts:
                            sub_counts[s_key] = {}
                        sub_counts[s_key][t_key] = sub_counts[s_key].get(t_key, 0.0) + w
                    elif path.is_insertion:
                        t_key = path.target_key
                        w = path.confidence
                        ins_counts[t_key] = ins_counts.get(t_key, 0.0) + w
                    elif path.is_deletion:
                        s_key = path.source_key
                        del_counts[s_key] = del_counts.get(s_key, 0.0) + 1.0

            mean_cost = total_cost_sum / max(1, total_alignments)

            # M-step
            new_probabilities: Dict[str, Dict[str, float]] = {}
            max_delta = 0.0
            all_src_keys = (
                set(src_vocab)
                | set(sub_counts.keys())
                | set(current_matrix.probabilities.keys())
            )

            for a in sorted(all_src_keys):
                prior_map = current_matrix.probabilities.get(a, {})
                obs_map = sub_counts.get(a, {})
                all_targets = set(obs_map.keys()) | set(prior_map.keys())
                if not all_targets:
                    continue

                effective_counts: Dict[str, float] = {
                    c: obs_map.get(c, 0.0) + self.alpha_prior * prior_map.get(c, 0.01)
                    for c in all_targets
                }
                total_a = sum(effective_counts.values())
                new_probabilities[a] = {}
                for c in all_targets:
                    p_new = effective_counts[c] / max(total_a, 1e-12)
                    new_probabilities[a][c] = p_new
                    old_p = prior_map.get(c, 0.0)
                    max_delta = max(max_delta, abs(p_new - old_p))

            # Re-estimate insertion probabilities
            new_ins_probs: Dict[str, float] = {}
            prior_ins = current_matrix.insertion_probabilities
            all_ins_c = set(ins_counts.keys()) | set(prior_ins.keys()) | set(tgt_vocab)
            eff_ins = {
                c: ins_counts.get(c, 0.0)
                + self.alpha_prior * prior_ins.get(c, 1.0 / max(1, len(all_ins_c)))
                for c in all_ins_c
            }
            tot_ins = sum(eff_ins.values())
            for c in all_ins_c:
                new_ins_probs[c] = eff_ins[c] / max(tot_ins, 1e-12)

            # Re-estimate deletion probabilities
            new_del_probs: Dict[str, float] = {}
            prior_del = current_matrix.deletion_probabilities
            for a in all_src_keys:
                tot_sub_a = sum(sub_counts.get(a, {}).values())
                tot_del_a = del_counts.get(a, 0.0)
                tot_a = tot_sub_a + tot_del_a
                if tot_a > 0:
                    new_del_probs[a] = (
                        tot_del_a + self.alpha_prior * prior_del.get(a, 0.1)
                    ) / (tot_a + self.alpha_prior)
                else:
                    new_del_probs[a] = prior_del.get(a, 0.1)

            last_delta = max_delta

            current_matrix = AcousticConfusionMatrix.create(
                model_id=model_id,
                probabilities=new_probabilities,
                insertion_probabilities=new_ins_probs,
                deletion_probabilities=new_del_probs,
                source_vocab=tuple(sorted(new_probabilities.keys())),
                target_vocab=tgt_vocab,
                prune_threshold=self.prune_threshold,
                iteration=iteration,
                metadata={
                    "iteration": iteration,
                    "delta": max_delta,
                    "mean_cost": mean_cost,
                },
            )

        # Pruning step
        pruned_probabilities: Dict[str, Dict[str, float]] = {}
        for a in sorted(current_matrix.probabilities.keys()):
            raw_map = current_matrix.probabilities.get(a, {})
            filtered = {c: p for c, p in raw_map.items() if p >= self.prune_threshold}
            if not filtered and raw_map:
                best_c, best_p = max(raw_map.items(), key=lambda kv: kv[1])
                filtered = {best_c: best_p}

            if filtered:
                sum_f = sum(filtered.values())
                pruned_probabilities[a] = {c: p / sum_f for c, p in filtered.items()}

        raw_ins = current_matrix.insertion_probabilities
        filtered_ins = {c: p for c, p in raw_ins.items() if p >= self.prune_threshold}
        if not filtered_ins and raw_ins:
            best_c, best_p = max(raw_ins.items(), key=lambda kv: kv[1])
            filtered_ins = {best_c: best_p}
        sum_ins_f = sum(filtered_ins.values())
        pruned_insertion = {c: p / sum_ins_f for c, p in filtered_ins.items()}

        pruned_deletion = {
            a: max(0.01, min(0.99, p))
            for a, p in current_matrix.deletion_probabilities.items()
        }

        final_matrix = AcousticConfusionMatrix.create(
            model_id=model_id,
            probabilities=pruned_probabilities,
            insertion_probabilities=pruned_insertion,
            deletion_probabilities=pruned_deletion,
            source_vocab=tuple(sorted(pruned_probabilities.keys())),
            target_vocab=tgt_vocab,
            prune_threshold=self.prune_threshold,
            iteration=self.num_iterations,
            metadata={
                "trained_on_pairs": len(paired_data),
                "num_iterations": self.num_iterations,
                "prune_threshold": self.prune_threshold,
                "final_delta": last_delta,
                "mean_cost": mean_cost,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return final_matrix


__all__ = [
    "ConfusionMatrixTrainer",
    "align_word_pair_generic",
    "bucket_by_duration",
    "ctc_prefix_beam_search",
    "sanitize_model_id",
]
