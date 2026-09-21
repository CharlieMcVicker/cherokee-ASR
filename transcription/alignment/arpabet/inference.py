# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.inference module.

Batched procedural inference and emissions caching for CherokeeASRModel over
single-word audio clips. Implements duration-sorted batching, greedy decoding
with phonotactic digraph grouping, CTC prefix beam search for Top-K hypotheses,
and durable emissions cache management.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

from transcription.alignment.arpabet.dataset import load_words_manifest
from transcription.alignment.arpabet.types import (
    CherokeeToken,
    InferenceCacheManifest,
    TopKHypothesis,
    WordInferenceCacheEntry,
    WordManifestEntry,
)
from transcription.alignment.phonotactics import tokenize_phonemes
from transcription.models.asr_model import CherokeeASRModel
from transcription.utils.orthography import Orthography

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sanitize_model_id(model_id: str) -> str:
    """
    Sanitizes a model identifier or repo path for safe filesystem cache storage.

    Replaces slashes, colons, spaces, and other non-alphanumeric characters
    (except hyphens, underscores, and periods) with underscores.

    Examples:
        >>> sanitize_model_id("charliemcvicker/cherokee-wav2vec2")
        'charliemcvicker_cherokee-wav2vec2'
        >>> sanitize_model_id("facebook/wav2vec2-base-960h:v1")
        'facebook_wav2vec2-base-960h_v1'
    """
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

    Minimizes wasted padding frames during batched ASR model forward passes on
    MPS or CPU devices.

    Args:
        manifest_entries: Sequence of items (e.g. WordManifestEntry).
        batch_size: Maximum batch size. Must be > 0.
        duration_key: Optional custom duration accessor callable. If None,
            inspects entry.duration or entry["duration"].

    Returns:
        List of batches, each containing up to batch_size entries sorted by duration.
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


# Sibilant clusters decomposed into canonical atomic Cherokee phonemes
CLUSTER_DECOMPOSITIONS: Dict[str, Tuple[Tuple[str, int, int], ...]] = {
    "hskw": (("hs", 0, 2), ("kw", 2, 4)),
    "hsts": (("hs", 0, 2), ("ts", 2, 4)),
    "hsk": (("hs", 0, 2), ("k", 2, 3)),
    "hst": (("hs", 0, 2), ("t", 2, 3)),
    "hsl": (("hs", 0, 2), ("l", 2, 3)),
    "slh": (("s", 0, 1), ("lh", 1, 3)),
}


def group_digraphs_with_confidences(
    text: str,
    char_confidences: Sequence[float],
) -> Tuple[Tuple[CherokeeToken, ...], Tuple[float, ...]]:
    """
    Groups Cherokee phonetic text into atomic CherokeeToken units using tokenize_phonemes.

    Multi-character phonemes (e.g. digraphs/trigraphs 'hs', 'th', 'kh', 'ts', 'tsh',
    'tl', 'tlh', 'lh', 'nh', 'wh', 'yh') are grouped into single CherokeeToken units.
    Their confidence scores are calculated as the arithmetic mean of the constituent
    characters' confidences. Sibilant clusters are resolved to their atomic phonemes
    (e.g. 'hsk' -> 'hs' + 'k').

    Args:
        text: Transcribed Cherokee phonetic string (TTH orthography).
        char_confidences: Sequence of per-character confidence scores matching text.

    Returns:
        Tuple of (grouped CherokeeToken tuple, averaged token confidences tuple).
    """
    if not text:
        return (), ()

    confs = list(char_confidences)
    if len(confs) < len(text):
        confs.extend([1.0] * (len(text) - len(confs)))

    tokens: List[CherokeeToken] = []
    token_confs: List[float] = []

    phonotactic_tokens = tokenize_phonemes(text)
    for tok in phonotactic_tokens:
        symbol = tok.symbol
        # Skip pure whitespace tokens between words if any
        if not symbol.strip():
            continue

        sym_lower = symbol.lower()
        if sym_lower in CLUSTER_DECOMPOSITIONS:
            for sub_sym, rel_s, rel_e in CLUSTER_DECOMPOSITIONS[sym_lower]:
                s_idx = tok.start_idx + rel_s
                e_idx = tok.start_idx + rel_e
                slice_c = confs[s_idx:e_idx]
                avg_c = float(sum(slice_c) / len(slice_c)) if slice_c else 0.0
                tokens.append(CherokeeToken(sub_sym, orthography=Orthography.TTH))
                token_confs.append(avg_c)
        else:
            slice_c = confs[tok.start_idx : tok.end_idx]
            avg_c = float(sum(slice_c) / len(slice_c)) if slice_c else 0.0
            tokens.append(CherokeeToken(symbol, orthography=Orthography.TTH))
            token_confs.append(avg_c)

    return tuple(tokens), tuple(token_confs)


def decode_greedy_with_confidences(
    logits: Union[torch.Tensor, np.ndarray],
    processor: Any,
) -> Tuple[str, Tuple[CherokeeToken, ...], Tuple[float, ...], float]:
    """
    Performs greedy CTC decoding and extracts character-level confidences,
    followed by digraph/trigraph grouping into CherokeeToken phonemes.

    Args:
        logits: 2D tensor or array of shape [T, V].
        processor: Wav2Vec2Processor instance.

    Returns:
        Tuple of (greedy_text, greedy_tokens, token_confidences, mean_confidence).
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
    else:
        max_val = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_val)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    if probs.ndim == 3:
        probs = probs.squeeze(0)

    T, _ = probs.shape
    if T == 0:
        return "", (), (), 0.0

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.vocab.get("[PAD]", 0)

    word_delimiter_token_id = getattr(
        processor.tokenizer, "word_delimiter_token_id", None
    )

    pred_ids = np.argmax(probs, axis=-1)

    chars: List[str] = []
    char_confs: List[float] = []

    prev_id = -1
    cur_run_probs: List[float] = []

    for t, pid in enumerate(pred_ids):
        if pid == prev_id:
            if pid != pad_id:
                cur_run_probs.append(float(probs[t, pid]))
        else:
            if cur_run_probs and prev_id != pad_id:
                peak_prob = max(cur_run_probs)
                if prev_id == word_delimiter_token_id:
                    token_str = " "
                else:
                    token_str = processor.decode([prev_id])
                if token_str:
                    for c in token_str:
                        chars.append(c)
                        char_confs.append(peak_prob)
            cur_run_probs = []
            if pid != pad_id:
                cur_run_probs.append(float(probs[t, pid]))
        prev_id = pid

    if cur_run_probs and prev_id != pad_id:
        peak_prob = max(cur_run_probs)
        if prev_id == word_delimiter_token_id:
            token_str = " "
        else:
            token_str = processor.decode([prev_id])
        if token_str:
            for c in token_str:
                chars.append(c)
                char_confs.append(peak_prob)

    raw_text = "".join(chars)
    # Strip whitespace cleanly while aligning confidences
    l_strip = len(raw_text) - len(raw_text.lstrip())
    r_strip = len(raw_text) - len(raw_text.rstrip())
    text = raw_text.strip()
    end_idx = len(char_confs) - r_strip if r_strip > 0 else len(char_confs)
    aligned_confs = char_confs[l_strip:end_idx]

    tokens, token_confs = group_digraphs_with_confidences(text, aligned_confs)
    mean_conf = float(np.mean(token_confs)) if token_confs else 0.0

    return text, tokens, token_confs, mean_conf


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
    logits: Union[torch.Tensor, np.ndarray],
    processor: Any,
    beam_width: int = 10,
    top_k: int = 3,
    prune_threshold: float = 1e-4,
) -> List[Tuple[int, str, float, Tuple[float, ...]]]:
    """
    CTC prefix beam search decoding for acoustic logits.

    Args:
        logits: CTC logits tensor or array of shape [T, V].
        processor: Wav2Vec2Processor instance with tokenizer vocabulary.
        beam_width: Number of beam search paths retained per frame.
        top_k: Number of ranked hypotheses to return.
        prune_threshold: Minimum probability threshold for candidate tokens per frame.

    Returns:
        List of tuples: (rank, decoded_text, sequence_logprob_score, char_confidences).
    """
    if isinstance(logits, torch.Tensor):
        log_probs = (
            torch.nn.functional.log_softmax(logits, dim=-1).detach().cpu().numpy()
        )
    else:
        max_val = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_val)
        log_probs = (
            logits - max_val - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
        )

    if log_probs.ndim == 3:
        log_probs = log_probs.squeeze(0)

    T, V = log_probs.shape
    if T == 0:
        return [(1, "", 0.0, ())]

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.vocab.get("[PAD]", 0)

    log_prune_threshold = math.log(prune_threshold)

    # State: prefix -> (p_blank, p_non_blank, char_confidences)
    current_beam: Dict[Tuple[int, ...], Tuple[float, float, Tuple[float, ...]]] = {
        (): (0.0, -float("inf"), ())
    }

    for t in range(T):
        next_beam: Dict[Tuple[int, ...], Tuple[float, float, Tuple[float, ...]]] = {}
        l_t = log_probs[t]

        # Select candidate tokens passing threshold
        c_candidates: List[int] = [pad_id]
        for c in range(V):
            if c != pad_id and l_t[c] >= log_prune_threshold:
                c_candidates.append(c)

        for p, (p_b, p_nb, confs) in current_beam.items():
            p_total = _log_add(p_b, p_nb)

            # 1. Blank transition
            new_p_b = p_total + float(l_t[pad_id])
            curr_b, curr_nb, curr_confs = next_beam.get(
                p, (-float("inf"), -float("inf"), confs)
            )
            next_beam[p] = (_log_add(curr_b, new_p_b), curr_nb, curr_confs)

            # 2. Non-blank transitions
            for c in c_candidates:
                if c == pad_id:
                    continue
                p_c = float(l_t[c])
                prob_c = math.exp(p_c)

                if len(p) > 0 and c == p[-1]:
                    # Repeated token:
                    # Extension from blank yields a distinct new token
                    new_p = p + (c,)
                    new_nb_from_b = p_b + p_c
                    new_confs = confs + (prob_c,)
                    cb, cnb, cc = next_beam.get(
                        new_p, (-float("inf"), -float("inf"), new_confs)
                    )
                    next_beam[new_p] = (cb, _log_add(cnb, new_nb_from_b), cc)

                    # Extension from non-blank collapses into existing token
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

        # Prune to beam_width
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

    results: List[Tuple[int, str, float, Tuple[float, ...]]] = []
    for rank, (prefix, (pb, pnb, confs)) in enumerate(sorted_results[:top_k], start=1):
        total_score = _log_add(pb, pnb)
        raw_decoded = processor.decode(list(prefix))
        l_strip = len(raw_decoded) - len(raw_decoded.lstrip())
        r_strip = len(raw_decoded) - len(raw_decoded.rstrip())
        decoded = raw_decoded.strip()
        end_idx = len(confs) - r_strip if r_strip > 0 else len(confs)
        aligned_confs = confs[l_strip:end_idx]
        results.append((rank, decoded, total_score, tuple(aligned_confs)))

    if not results:
        results.append((1, "", 0.0, ()))

    return results


def extract_top_k_hypotheses(
    logits: Union[torch.Tensor, np.ndarray],
    processor: Any,
    beam_width: int = 10,
    top_k: int = 3,
    prune_threshold: float = 1e-4,
) -> Tuple[TopKHypothesis, ...]:
    """
    Extracts Top-K beam search hypotheses from CTC logits, grouping digraphs into
    CherokeeToken phonemes with averaged confidences.

    Args:
        logits: CTC logits tensor or array [T, V].
        processor: Wav2Vec2Processor instance.
        beam_width: Beam width.
        top_k: Number of hypotheses to return (default: 3).
        prune_threshold: CTC frame pruning threshold.

    Returns:
        Tuple of immutable TopKHypothesis objects.
    """
    raw_hyps = ctc_prefix_beam_search(
        logits=logits,
        processor=processor,
        beam_width=beam_width,
        top_k=top_k,
        prune_threshold=prune_threshold,
    )

    hypotheses: List[TopKHypothesis] = []
    for rank, text, score, char_confs in raw_hyps:
        tokens, tok_confs = group_digraphs_with_confidences(text, char_confs)
        hypotheses.append(
            TopKHypothesis(
                rank=rank,
                text=text,
                score=score,
                tokens=tokens,
                token_confidences=tok_confs,
            )
        )

    return tuple(hypotheses)


def run_model_inference_on_manifest(
    model: CherokeeASRModel,
    manifest: List[WordManifestEntry],
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    cache_dir: Union[Path, str] = "data/arpabet_alignment/cache",
    force_recompute: bool = False,
    show_progress: bool = True,
) -> InferenceCacheManifest:
    """
    Runs batched forward pass inference over a word manifest using CherokeeASRModel,
    tokenizes Cherokee emissions into atomic Cherokee phonemes with digraph grouping,
    and caches the results to data/arpabet_alignment/cache/{sanitized_model_id}_emissions.json.

    If cache exists and force_recompute is False, bypasses forward pass completely
    and loads the cached InferenceCacheManifest.

    Args:
        model: Initialized CherokeeASRModel instance.
        manifest: List of WordManifestEntry instances.
        batch_size: Batch size for duration-sorted forward pass.
        device: Optional target PyTorch device ('mps', 'cpu', 'cuda').
        cache_dir: Directory where emissions cache JSON files are stored.
        force_recompute: If True, executes inference even if cache exists.
        show_progress: If True and tqdm is installed, displays progress bar.

    Returns:
        InferenceCacheManifest with cached or computed emissions.
    """
    if device is not None:
        model.to(device)

    model_id = getattr(model, "model_name", None)
    if not model_id and hasattr(model, "model") and hasattr(model.model, "config"):
        model_id = getattr(model.model.config, "_name_or_path", None)
    if not model_id:
        model_id = "cherokee_asr_model"

    sanitized_id = sanitize_model_id(model_id)
    cache_path = Path(cache_dir) / f"{sanitized_id}_emissions.json"

    # Check cache hit
    if not force_recompute and cache_path.exists():
        logger.info(
            "Loading cached emissions from %s (bypassing forward pass)",
            cache_path,
        )
        return InferenceCacheManifest.load(cache_path)

    logger.info(
        "Running forward pass inference on %d clips with model '%s' (batch_size=%d, device=%s)...",
        len(manifest),
        model_id,
        batch_size,
        model.device,
    )

    if not manifest:
        empty_manifest = InferenceCacheManifest(
            model_id=model_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            entries=(),
            metadata={"total_clips": 0, "batch_size": batch_size},
        )
        empty_manifest.save(cache_path)
        return empty_manifest

    # Duration-sorted bucketing
    batches = bucket_by_duration(manifest, batch_size=batch_size)
    cache_entries: List[WordInferenceCacheEntry] = []

    iterator = batches
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc=f"ASR Inference ({sanitized_id[:25]})")
        except ImportError:
            iterator = batches

    for batch in iterator:
        audio_paths = [e.audio_path for e in batch]
        logits_list = model.get_logits_batch(audio_paths, batch_size=len(batch))

        for entry, logits in zip(batch, logits_list):
            greedy_text, greedy_tokens, token_confs, mean_conf = (
                decode_greedy_with_confidences(logits, model.processor)
            )
            top_hyps = extract_top_k_hypotheses(
                logits, model.processor, beam_width=10, top_k=3
            )

            cache_entry = WordInferenceCacheEntry(
                clip_id=entry.clip_id,
                word=entry.word,
                greedy_tokens=greedy_tokens,
                greedy_text=greedy_text,
                token_confidences=token_confs,
                top_hypotheses=top_hyps,
                mean_confidence=mean_conf,
                duration=entry.duration,
            )
            cache_entries.append(cache_entry)

    # Reorder entries to preserve original manifest ordering
    entries_by_id = {e.clip_id: e for e in cache_entries}
    ordered_entries = [
        entries_by_id[m.clip_id] for m in manifest if m.clip_id in entries_by_id
    ]

    manifest_obj = InferenceCacheManifest(
        model_id=model_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(ordered_entries),
        metadata={
            "total_clips": len(ordered_entries),
            "batch_size": batch_size,
            "device": str(model.device),
        },
    )

    manifest_obj.save(cache_path)
    logger.info(
        "Successfully saved %d emission entries to %s",
        len(ordered_entries),
        cache_path,
    )
    return manifest_obj


def main() -> None:
    """CLI entrypoint for batch Cherokee ASR inference runner."""
    parser = argparse.ArgumentParser(
        description="Run batched Cherokee ASR inference and emissions caching on single-word clips"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/arpabet_alignment/words_manifest.json",
        help="Path to input words_manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for duration-sorted forward pass",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/arpabet_alignment/cache",
        help="Target emissions cache directory",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if emissions cache exists",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="charliemcvicker/length-only-20260704-155307-asr-cherokee-colon",
        help="Hugging Face repo or directory path of the Cherokee ASR model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="76e62140955f4738abdab345ea34068b02d8d2a2",
        help="Model revision/commit hash (defaults to toneless pre-Bible baseline)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (mps, cpu, cuda). Auto-selects mps if available.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    resolved_device = args.device
    if resolved_device is None:
        resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(
        f"Loading Cherokee ASR model '{args.repo}' (revision: {args.revision}) on device '{resolved_device}'..."
    )
    if args.repo:
        model = CherokeeASRModel.from_pretrained(
            path_or_repo=args.repo,
            revision=args.revision,
            device=resolved_device,
        )
    else:
        model = CherokeeASRModel.get_best_model(device=resolved_device)

    print(f"Loading manifest from '{args.manifest}'...")
    manifest = load_words_manifest(args.manifest)
    print(f"Loaded {len(manifest)} manifest entries.")

    result = run_model_inference_on_manifest(
        model=model,
        manifest=manifest,
        batch_size=args.batch_size,
        device=resolved_device,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        show_progress=True,
    )
    print(
        f"Inference complete: {len(result)} entries cached for model '{result.model_id}'."
    )


if __name__ == "__main__":
    main()
