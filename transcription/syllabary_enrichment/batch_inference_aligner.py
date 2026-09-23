# -*- coding: utf-8 -*-
"""
batch_inference_aligner.py

Runs batch inference on a data manifest (JSON/JSONL) using CherokeeASRModel / infer_emissions_batch
or cached emitted predictions, then aligns ground truth Cherokee Syllabary
characters against emitted ASR text using character level alignment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from transcription.cherokee.models.loader import CherokeeASRModel
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
)
from transcription.utils.model_utils import get_best_model_config


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Load data manifest from JSON or JSONL file.

    Supported record fields:
      - audio_filepath / audio_path / file_path / path
      - syllabary_text / cherokee_text / text / syllabary
      - emitted_text (optional, precomputed)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Data manifest not found: {manifest_path}")

    records = []
    if manifest_path.endswith(".jsonl"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON at line {line_idx + 1} of {manifest_path}: {e}"
                        )
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("records", data.get("data", [data]))
            else:
                raise ValueError(
                    f"Unexpected JSON root type in {manifest_path}: {type(data)}"
                )

    normalized_records = []
    for r in records:
        record = dict(r)
        audio_path = (
            record.get("audio_filepath")
            or record.get("audio_path")
            or record.get("file_path")
            or record.get("path")
            or ""
        )
        record["audio_filepath"] = audio_path

        syllabary_text = (
            record.get("syllabary_text")
            or record.get("cherokee_text")
            or record.get("text")
            or record.get("syllabary")
            or ""
        )
        record["syllabary_text"] = syllabary_text

        emitted_text = record.get("emitted_text", None)
        if emitted_text is not None:
            record["emitted_text"] = str(emitted_text).strip()

        aligned_pairs = record.get("aligned_pairs", None)
        if aligned_pairs is not None:
            record["aligned_pairs"] = aligned_pairs

        normalized_records.append(record)

    return normalized_records


def run_batch_inference(
    records: List[Dict[str, Any]],
    checkpoint: str,
    processor_path: Optional[str] = None,
    revision: Optional[str] = None,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[str]:
    """
    Execute batched ASR inference using CherokeeASRModel to generate emitted_text predictions.

    Args:
        records: List of manifest records containing 'audio_filepath'.
        checkpoint: Model path or HF repo ID.
        processor_path: Processor path or HF repo ID.
        revision: Commit hash, branch, or tag.
        batch_size: Batch size.
        num_workers: Ignored / kept for interface compatibility.
        hf_token: HuggingFace hub token.

    Returns:
        List of emitted_text strings corresponding 1-to-1 with input records.
    """
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    asr_model = CherokeeASRModel.from_pretrained(
        path_or_repo=checkpoint,
        revision=revision,
        processor_path=processor_path,
        token=token,
    )

    audio_paths = [r.get("audio_filepath", "") for r in records]
    valid_paths = [p for p in audio_paths if p and os.path.exists(p)]

    if not valid_paths:
        return ["" for _ in records]

    # Map audio paths to predictions
    outputs = asr_model.infer_batch(valid_paths, batch_size=batch_size)
    emissions_map: Dict[str, str] = {
        path: out.decode_greedy() for path, out in zip(valid_paths, outputs)
    }

    emitted_texts = [emissions_map.get(p, "") for p in audio_paths]
    return emitted_texts


def process_batch_inference_and_alignment(
    manifest_path: str,
    output_cache_path: str,
    force_recompute: bool = False,
    checkpoint: Optional[str] = None,
    processor_path: Optional[str] = None,
    revision: Optional[str] = None,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process batch inference and alignment with disk caching.
    """
    if os.path.exists(output_cache_path) and not force_recompute:
        with open(output_cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return cached_data

    records = load_manifest(manifest_path)

    needs_inference = [
        r for r in records if "emitted_text" not in r or r["emitted_text"] is None
    ]

    if needs_inference:
        best_cfg = get_best_model_config()
        ckpt = checkpoint or best_cfg.get("repo", "charliemcvicker/asr-cherokee")
        rev = revision or best_cfg.get("revision", "5464d15")
        proc = processor_path or ckpt

        emitted_preds = run_batch_inference(
            records=records,
            checkpoint=ckpt,
            processor_path=proc,
            revision=rev,
            batch_size=batch_size,
            num_workers=num_workers,
            hf_token=hf_token,
        )

        for rec, pred in zip(records, emitted_preds):
            if "emitted_text" not in rec or rec["emitted_text"] is None:
                rec["emitted_text"] = pred

    # Perform character-level alignment
    for rec in records:
        syllabary = rec.get("syllabary_text", "")
        emitted = rec.get("emitted_text", "")
        aligned_pairs = align_character_syllable(syllabary, emitted)
        rec["aligned_pairs"] = aligned_pairs

    out_dir = os.path.dirname(output_cache_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return records
