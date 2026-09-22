# -*- coding: utf-8 -*-
"""
manifest.py

Serialization exporters for alignment manifests and debug JSON logs.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from transcription.alignment.models import AlignmentOutput, WordInterval


def serialize_word_interval(w: WordInterval) -> Dict[str, Any]:
    """Helper to convert WordInterval into JSON serializable dict."""
    w_dict: Dict[str, Any] = {
        "word": w.word,
        "start": w.start_sec,
        "end": w.end_sec,
        "confidence": w.confidence,
        "flagged": w.flagged,
    }
    if w.emitted_word:
        w_dict["emitted_word"] = w.emitted_word
    return w_dict


def export_manifest(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "alignment_manifest.json",
    source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_word_tiers: Optional[Mapping[str, Sequence[WordInterval]]] = None,
) -> str:
    """
    Generates manifest JSON matching project schema, including optional additional word tiers.

    Args:
        alignment: AlignmentOutput object.
        output_dir: Directory path where manifest JSON should be saved.
        filename: Name of the manifest JSON output file.
        source_metadata: Optional chunk metadata dictionary.
        additional_word_tiers: Optional mapping of tier name to Sequence of WordIntervals.

    Returns:
        Path to the generated JSON manifest file.
    """
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), filename)

    manifest_lines = []
    meta_lookup = source_metadata or {}

    tier_chunk_words: Dict[str, List[List[WordInterval]]] = {}
    if additional_word_tiers:
        total_chunk_words = sum(len(c.words) for c in alignment.aligned_chunks)
        for tier_name, tier_words in additional_word_tiers.items():
            tier_word_list = list(tier_words)
            if len(tier_word_list) == total_chunk_words:
                cur = 0
                chunk_lists: List[List[WordInterval]] = []
                for c in alignment.aligned_chunks:
                    chunk_len = len(c.words)
                    chunk_lists.append(tier_word_list[cur : cur + chunk_len])
                    cur += chunk_len
                tier_chunk_words[tier_name] = chunk_lists
            else:
                chunk_lists = []
                for c in alignment.aligned_chunks:
                    matched = [
                        w
                        for w in tier_word_list
                        if (
                            w.start_sec >= c.start_sec - 0.05
                            and w.end_sec <= c.end_sec + 0.05
                        )
                        or (w.start_sec < c.end_sec and w.end_sec > c.start_sec)
                    ]
                    chunk_lists.append(matched)
                tier_chunk_words[tier_name] = chunk_lists

    for c_idx, c in enumerate(alignment.aligned_chunks):
        word_objs = []
        for w_idx, w in enumerate(c.words):
            w_dict = serialize_word_interval(w)
            rec_tier = tier_chunk_words.get("Reconciled Words") or tier_chunk_words.get(
                "Reconciled Transcriptions"
            )
            if rec_tier:
                rec_words_for_chunk = rec_tier[c_idx]
                if w_idx < len(rec_words_for_chunk):
                    w_dict["reconciled_word"] = rec_words_for_chunk[w_idx].word
            word_objs.append(w_dict)

        chunk_meta = meta_lookup.get(c.chunk_id, {})
        line_dict: Dict[str, Any] = {
            "line_id": c.chunk_id,
            "cherokee_syllabary": chunk_meta.get(
                "cherokee_syllabary", chunk_meta.get("cherokee", "")
            ),
            "text": chunk_meta.get("text", ""),
            "english": chunk_meta.get("english", ""),
            "start": c.start_sec,
            "end": c.end_sec,
            "cer": c.distance_score,
            "emitted_text": c.emitted_text,
            "words": word_objs,
        }

        if additional_word_tiers:
            chunk_additional_tiers: Dict[str, List[Dict[str, Any]]] = {}
            for tier_name in additional_word_tiers:
                tier_words_for_chunk = tier_chunk_words[tier_name][c_idx]
                chunk_additional_tiers[tier_name] = [
                    serialize_word_interval(tw) for tw in tier_words_for_chunk
                ]

            line_dict["additional_word_tiers"] = chunk_additional_tiers
            rec_chunk = chunk_additional_tiers.get(
                "Reconciled Words"
            ) or chunk_additional_tiers.get("Reconciled Transcriptions")
            if rec_chunk is not None:
                line_dict["reconciled_words"] = rec_chunk

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

    data: Dict[str, Any] = {
        "audio_source": alignment.source_id,
        "metrics": metrics_dict,
        "lines": manifest_lines,
    }

    if additional_word_tiers:
        top_additional_tiers: Dict[str, List[Dict[str, Any]]] = {}
        for tier_name, tier_words in additional_word_tiers.items():
            top_additional_tiers[tier_name] = [
                serialize_word_interval(tw) for tw in tier_words
            ]
        data["additional_word_tiers"] = top_additional_tiers
        rec_top = top_additional_tiers.get(
            "Reconciled Words"
        ) or top_additional_tiers.get("Reconciled Transcriptions")
        if rec_top is not None:
            data["reconciled_words"] = rec_top

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def export_debug_json(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "alignment_debug.json",
) -> str:
    """
    Writes raw tokens, counts, and metrics JSON into output_dir.

    Args:
        alignment: AlignmentOutput object.
        output_dir: Directory path where debug JSON should be saved.
        filename: Name of the debug JSON output file.

    Returns:
        Path to the generated debug JSON file.
    """
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), filename)

    raw_toks = [
        {
            "word": t.word,
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "confidence": t.confidence,
        }
        for t in (alignment.raw_tokens or [])
    ]

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

    debug_data = {
        "audio_source": alignment.source_id,
        "raw_tokens": raw_toks,
        "aligned_chunks_count": len(alignment.aligned_chunks),
        "metrics": metrics_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    return output_path
