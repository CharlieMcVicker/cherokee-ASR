# -*- coding: utf-8 -*-
"""
batch_inference_aligner.py

Runs batch inference on a data manifest (JSON/JSONL) using Wav2Vec2 batch GPU capabilities
(transcription.inference.batch) or cached emitted predictions, then aligns ground truth Cherokee Syllabary
characters against emitted ASR text using character level alignment (transcription.syllabary_enrichment.alignment_engine).

Results are persisted to disk cache manifest JSON by default with a --force-recompute flag to reload existing caches.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import time
import glob
import multiprocessing
from typing import List, Dict, Any, Optional


import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transcription.models.asr_model import CherokeeASRModel
from transcription.utils.model_utils import get_best_model_config


from transcription.inference.infer import (
    TARGET_SAMPLE_RATE,
    load_and_preprocess_audio,
)
from transcription.inference.batch import init_worker, decode_worker
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
)


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
                            f"Error parsing JSONL at line {line_idx + 1}: {e}"
                        )
    elif manifest_path.endswith(".csv"):
        import csv

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                if "records" in data:
                    records = data["records"]
                elif "data" in data:
                    records = data["data"]
                else:
                    records = [data]
            else:
                raise ValueError("JSON content must be a list or dict object.")

    # Normalize record field keys
    normalized_records = []
    for item in records:
        audio_path = (
            item.get("audio_filepath")
            or item.get("audio_path")
            or item.get("file_path")
            or item.get("path")
            or ""
        )
        syllabary_text = (
            item.get("syllabary_text")
            or item.get("cherokee_text")
            or item.get("syllabary")
            or item.get("text")
            or ""
        )
        emitted_text = item.get("emitted_text", None)
        aligned_pairs = item.get("aligned_pairs", None)

        record = dict(item)
        record["audio_filepath"] = audio_path
        record["syllabary_text"] = syllabary_text
        if emitted_text is not None:
            record["emitted_text"] = emitted_text
        if aligned_pairs is not None:
            record["aligned_pairs"] = aligned_pairs

        normalized_records.append(record)

    return normalized_records


def run_batch_inference(
    records: List[Dict[str, Any]],
    checkpoint: str,
    processor_path: str,
    revision: str,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[str]:
    """
    Execute batched GPU ASR inference using batch.py capabilities to generate emitted_text predictions.

    Args:
        records: List of manifest records containing 'audio_filepath'.
        checkpoint: Model path or HF repo ID.
        processor_path: Processor path or HF repo ID.
        revision: Commit hash, branch, or tag.
        batch_size: GPU batch size.
        num_workers: Multiprocessing CPU workers for decoding.
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
    model = asr_model.model
    processor = asr_model.processor
    device = asr_model.device

    # Prepare audio metadata and track record indices
    valid_audios = []
    for idx, rec in enumerate(records):
        audio_path = rec["audio_filepath"]
        filename = os.path.basename(audio_path)
        if not audio_path or not os.path.exists(audio_path):
            valid_audios.append(
                {
                    "record_idx": idx,
                    "audio_path": audio_path,
                    "filename": filename,
                    "length": 0,
                    "invalid": True,
                    "emitted_text": "",
                }
            )
            continue

        try:
            if os.path.getsize(audio_path) == 0:
                valid_audios.append(
                    {
                        "record_idx": idx,
                        "audio_path": audio_path,
                        "filename": filename,
                        "length": 0,
                        "invalid": True,
                        "emitted_text": "",
                    }
                )
                continue
            info = sf.info(audio_path)
            if info.frames < 400:
                valid_audios.append(
                    {
                        "record_idx": idx,
                        "audio_path": audio_path,
                        "filename": filename,
                        "length": info.frames,
                        "invalid": True,
                        "emitted_text": "",
                    }
                )
                continue
            valid_audios.append(
                {
                    "record_idx": idx,
                    "audio_path": audio_path,
                    "filename": filename,
                    "length": info.frames,
                    "invalid": False,
                }
            )
        except Exception as e:
            valid_audios.append(
                {
                    "record_idx": idx,
                    "audio_path": audio_path,
                    "filename": filename,
                    "length": 0,
                    "invalid": True,
                    "emitted_text": "",
                }
            )

    to_process = [a for a in valid_audios if not a.get("invalid", False)]

    # Sort by length for efficient batching
    to_process.sort(key=lambda x: x["length"])

    batches = [
        to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)
    ]

    workers_count = num_workers or multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
        processes=workers_count,
        initializer=init_worker,
        initargs=(processor_path, token, revision),
    )

    results_map: Dict[int, str] = {}

    # Handle invalid files directly
    for a in valid_audios:
        if a.get("invalid", False):
            results_map[a["record_idx"]] = a.get("emitted_text", "")

    def collect_callback(res):
        results_map[res["index"]] = res["greedy_raw"]

    for batch in batches:
        speech_list = []
        for item in batch:
            try:
                speech = load_and_preprocess_audio(
                    item["audio_path"], TARGET_SAMPLE_RATE
                )
                item["speech"] = speech
                speech_list.append(speech)
            except Exception as e:
                speech = np.zeros(16000, dtype=np.float32)
                item["speech"] = speech
                speech_list.append(speech)

        inputs = processor(
            speech_list,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        try:
            with torch.no_grad():
                if attention_mask is not None:
                    batch_logits = model(
                        input_values, attention_mask=attention_mask
                    ).logits
                else:
                    batch_logits = model(input_values).logits
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "mps" in err_str:
                # MPS or OOM fallback to CPU/sequential
                if device == "mps":
                    device = "cpu"
                    from typing import cast

                    cast(Any, model).to(device)
                input_values = input_values.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.no_grad():
                    if attention_mask is not None:
                        batch_logits = model(
                            input_values, attention_mask=attention_mask
                        ).logits
                    else:
                        batch_logits = model(input_values).logits
            else:
                raise e

        for item_idx, item in enumerate(batch):
            input_len = len(item["speech"])
            logit_len = int(model._get_feat_extract_output_lengths(input_len))
            logits_np = batch_logits[item_idx, :logit_len].detach().cpu().numpy().copy()
            item_data = (
                item["record_idx"],
                item["filename"],
                item["audio_path"],
                logits_np,
            )
            pool.apply_async(decode_worker, (item_data,), callback=collect_callback)

        # Free speech arrays
        for item in batch:
            item.pop("speech", None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pool.close()
    pool.join()

    # Reconstruct emitted texts in original record order
    emitted_texts = [results_map.get(i, "") for i in range(len(records))]
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

    1. Checks if output_cache_path exists and force_recompute is False.
       If cached, loads and returns existing records.
    2. Ingests data manifest.
    3. Executes batch GPU inference if emitted_text is not provided.
    4. Runs character alignment align_character_syllable().
    5. Saves result to disk cache manifest JSON.
    """
    if not force_recompute and os.path.exists(output_cache_path):
        print(f"Loading cached alignment manifest from {output_cache_path}...")
        with open(output_cache_path, "r", encoding="utf-8") as f:
            cached_records = json.load(f)
        return cached_records

    print(f"Processing data manifest from {manifest_path}...")
    records = load_manifest(manifest_path)

    # Check if we need GPU inference for missing emitted_texts
    needs_inference = any(
        "emitted_text" not in rec or rec["emitted_text"] is None for rec in records
    )

    if needs_inference:
        model_config = get_best_model_config()
        resolved_checkpoint = checkpoint or model_config["repo"]
        resolved_processor_path = processor_path or model_config["repo"]
        resolved_revision = revision or model_config["revision"]

        print("Executing batched GPU ASR inference...")
        emitted_texts = run_batch_inference(
            records=records,
            checkpoint=resolved_checkpoint,
            processor_path=resolved_processor_path,
            revision=resolved_revision,
            batch_size=batch_size,
            num_workers=num_workers,
            hf_token=hf_token,
        )
        for rec, emitted in zip(records, emitted_texts):
            if "emitted_text" not in rec or rec["emitted_text"] is None:
                rec["emitted_text"] = emitted

    # Perform character alignment for each record
    print("Performing character & syllable alignment...")
    for rec in records:
        syllabary_text = rec.get("syllabary_text", "")
        emitted_text = rec.get("emitted_text", "")
        aligned_pairs = align_character_syllable(syllabary_text, emitted_text)
        rec["aligned_pairs"] = aligned_pairs

    # Save to disk cache manifest JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_cache_path)), exist_ok=True)
    with open(output_cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"Successfully processed and cached {len(records)} records to {output_cache_path}"
    )
    return records


def main():
    model_config = get_best_model_config()
    default_repo = model_config["repo"]
    default_revision = model_config["revision"]

    parser = argparse.ArgumentParser(
        description="Batch ASR Inference and Syllabary Alignment with Disk Caching"
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to input data manifest JSON or JSONL file.",
    )
    parser.add_argument(
        "--output-cache",
        type=str,
        default="data/results/aligned_manifest_cache.json",
        help="Path to output cache manifest JSON file (default: data/results/aligned_manifest_cache.json).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=False,
        help="Force recomputation of inference and alignment even if cache exists (default: False).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_repo,
        help="Path to model checkpoint or HF Hub repo ID.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=default_repo,
        help="Path to processor or HF Hub repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=default_revision,
        help="HF Hub revision or tag.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU forward pass (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for decoding.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token.",
    )

    args = parser.parse_args()

    process_batch_inference_and_alignment(
        manifest_path=args.manifest_path,
        output_cache_path=args.output_cache,
        force_recompute=args.force_recompute,
        checkpoint=args.checkpoint,
        processor_path=args.processor,
        revision=args.revision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
