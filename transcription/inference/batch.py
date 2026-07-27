#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch.py

Runs speech-to-text inference on a directory of WAV files using the fine-tuned Wav2Vec2 model.
Optimized using batched GPU inference and multiprocessed CPU CTC decoding with detailed confidence scores.
"""

import os

# Enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Any
import argparse
import sys
import glob
import csv
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import time
import multiprocessing
from tqdm import tqdm

from transcription.utils.model_utils import get_best_model_config
from transcription.inference.infer import (
    TARGET_SAMPLE_RATE,
    load_and_preprocess_audio,
    calculate_word_confidences,
    greedy_inference,
)

# Global variables in the worker processes to avoid serializing the processor objects
global_processor = None


def init_worker(processor_path, token, revision):
    """
    Initialize global processor in worker processes once.
    This avoids pickle serialization overhead.
    """
    global global_processor
    # Restrict internal Torch threading in workers to prevent CPU oversubscription
    torch.set_num_threads(1)

    from transformers import Wav2Vec2Processor

    global_processor = Wav2Vec2Processor.from_pretrained(
        processor_path, token=token, revision=revision
    )


def decode_worker(item_data):
    """
    Worker function to decode logits and compute detailed confidences.
    item_data is a tuple: (index, filename, audio_path, logits_np)
    """
    global global_processor
    import numpy as np
    import torch
    import json
    from transcription.inference.infer import (
        calculate_word_confidences,
        greedy_inference,
    )

    idx, filename, audio_path, logits_np = item_data

    logits_tensor = torch.tensor(logits_np)
    res = greedy_inference(logits_tensor, global_processor)
    if isinstance(res, list):
        res = res[0]
    greedy_raw = str(res["text"])
    greedy_confidence = float(res["confidence"])

    # Calculate detailed character/word confidences
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1).numpy()
    pred_ids = np.argmax(probs, axis=-1)
    words_details = calculate_word_confidences(probs, pred_ids, global_processor)
    word_confidences_json = json.dumps(words_details, ensure_ascii=False)

    return {
        "index": idx,
        "filename": filename,
        "audio_path": audio_path,
        "greedy_raw": greedy_raw,
        "greedy_confidence": greedy_confidence,
        "word_confidences": word_confidences_json,
    }


def main():
    model_config = get_best_model_config()
    default_repo = model_config["repo"]
    default_revision = model_config["revision"]

    parser = argparse.ArgumentParser(
        description="Run Wav2Vec2 ASR batch inference on a directory of audio files."
    )
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the directory containing WAV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_repo,
        help="Path to model checkpoint or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=default_repo,
        help="Path to saved processor or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Hub authentication token.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=default_revision,
        help="Specific Hugging Face Hub commit hash, branch, or tag.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/batch_inference_results.csv",
        help="Path to the output CSV file to save results.",
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
        help="Number of worker processes for parallel decoding (default: CPU count).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir_path):
        print(f"Error: Directory '{args.dir_path}' not found.", flush=True)
        sys.exit(1)

    # Find WAV files
    wav_files = []
    for ext in ("*.wav", "*.WAV"):
        wav_files.extend(glob.glob(os.path.join(args.dir_path, ext)))

    wav_files = sorted(list(set(wav_files)))
    if not wav_files:
        print(f"No WAV files found in '{args.dir_path}'.", flush=True)
        sys.exit(0)

    print(f"Found {len(wav_files)} WAV files to process.", flush=True)

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if os.path.exists(args.processor):
        print(f"Loading processor from local path: {args.processor}...", flush=True)
    else:
        print(
            f"Loading processor from Hugging Face Hub: {args.processor} (revision: {args.revision})...",
            flush=True,
        )
    processor: Any = Wav2Vec2Processor.from_pretrained(
        args.processor, token=token, revision=args.revision
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading model from local path: {args.checkpoint}...", flush=True)
    else:
        print(
            f"Loading model from Hugging Face Hub: {args.checkpoint} (revision: {args.revision})...",
            flush=True,
        )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint, token=token, revision=args.revision
    )
    model.eval()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)
    model.to(device)  # type: ignore

    # Read audio metadata for sorting (lazy loading to minimize VRAM/RAM)
    loaded_audios = []
    print("Reading audio metadata for sorting...", flush=True)
    audio_load_start = time.time()
    for audio_path in wav_files:
        filename = os.path.basename(audio_path)
        try:
            if os.path.getsize(audio_path) == 0:
                print(f"Skipping 0-length file: {filename}", flush=True)
                continue
            info = sf.info(audio_path)
            if info.frames < 400:
                print(
                    f"Skipping file that is too short ({info.frames} frames): {filename}",
                    flush=True,
                )
                continue
            loaded_audios.append(
                {"audio_path": audio_path, "filename": filename, "length": info.frames}
            )
        except Exception as e:
            print(f"Error reading metadata for {filename}: {e}", flush=True)

    if not loaded_audios:
        print("No audio files successfully loaded. Exiting.", flush=True)
        sys.exit(0)

    # Sort by length
    loaded_audios.sort(key=lambda x: x["length"])
    print(
        f"Successfully loaded {len(loaded_audios)} files in {time.time() - audio_load_start:.2f}s.",
        flush=True,
    )

    # Create batches
    batches = [
        loaded_audios[i : i + args.batch_size]
        for i in range(0, len(loaded_audios), args.batch_size)
    ]

    # Initialize multiprocessing Pool for parallel decoding
    num_workers = args.num_workers or multiprocessing.cpu_count()
    print(
        f"Initializing multiprocessing pool with {num_workers} workers...", flush=True
    )

    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.processor, token, args.revision),
    )

    # Initialize CSV headers
    headers = [
        "file_path",
        "filename",
        "greedy_transcription",
        "greedy_confidence",
        "word_confidences",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    import threading

    csv_lock = threading.Lock()

    # Bounded semaphore to prevent IPC queue explosion and system OOM/segfault
    max_queue = threading.Semaphore(args.batch_size * 5)

    def write_result_callback(res):
        try:
            with csv_lock:
                with open(args.output, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    row = [
                        res["audio_path"],
                        res["filename"],
                        res["greedy_raw"],
                        f"{res['greedy_confidence']:.4f}",
                        res.get("word_confidences", "[]"),
                    ]
                    writer.writerow(row)
        finally:
            max_queue.release()

    global_idx = 0

    gpu_start_time = time.time()
    print(
        f"Running batched GPU inference and streaming decoding (batch size: {args.batch_size}, total batches: {len(batches)})...",
        flush=True,
    )

    pbar = tqdm(total=len(loaded_audios), desc="Transcribing", unit="file")
    for batch_idx, batch in enumerate(batches, 1):
        speech_list = []
        for item in batch:
            try:
                speech = load_and_preprocess_audio(
                    item["audio_path"], TARGET_SAMPLE_RATE
                )
                item["speech"] = speech
                speech_list.append(speech)
            except Exception as e:
                print(
                    f"Error reading audio {item['filename']} during batching: {e}",
                    flush=True,
                )
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
            is_oom = "out of memory" in err_str or (
                hasattr(torch.cuda, "OutOfMemoryError")
                and isinstance(e, torch.cuda.OutOfMemoryError)
            )
            is_cudnn_err = "unable to find an engine" in err_str or "cudnn" in err_str

            if is_oom or is_cudnn_err:
                reason = "OOM" if is_oom else "cuDNN error"
                print(
                    f"  {reason} on batch {batch_idx}. Falling back to sequential inference for this batch...",
                    flush=True,
                )

                # Free large tensors to ensure empty_cache succeeds
                for var_name in ("input_values", "attention_mask"):
                    locals().pop(var_name, None)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for item_idx, item in enumerate(batch):
                    single_inputs = processor(
                        [item["speech"]],
                        sampling_rate=TARGET_SAMPLE_RATE,
                        padding=True,
                        return_tensors="pt",
                    )
                    single_input_values = single_inputs.input_values.to(device)
                    single_attention_mask = getattr(
                        single_inputs, "attention_mask", None
                    )
                    if single_attention_mask is not None:
                        single_attention_mask = single_attention_mask.to(device)

                    try:
                        with torch.no_grad():
                            with torch.backends.cudnn.flags(enabled=False):
                                if single_attention_mask is not None:
                                    single_logits = model(
                                        single_input_values,
                                        attention_mask=single_attention_mask,
                                    ).logits
                                else:
                                    single_logits = model(single_input_values).logits

                        input_len = len(item["speech"])
                        logit_len = int(
                            model._get_feat_extract_output_lengths(input_len)
                        )
                        logits_np = (
                            single_logits[0, :logit_len].detach().cpu().numpy().copy()
                        )
                        item_data = (
                            global_idx,
                            item["filename"],
                            item["audio_path"],
                            logits_np,
                        )
                        max_queue.acquire()
                        pool.apply_async(
                            decode_worker, (item_data,), callback=write_result_callback
                        )
                        global_idx += 1
                    except Exception as seq_e:
                        print(
                            f"  Fatal OOM on {item['filename']} even with batch_size=1. Skipping file...",
                            flush=True,
                        )
                        empty_np = np.zeros((1, 32), dtype=np.float32)
                        item_data = (
                            global_idx,
                            item["filename"],
                            item["audio_path"],
                            empty_np,
                        )
                        max_queue.acquire()
                        pool.apply_async(
                            decode_worker, (item_data,), callback=write_result_callback
                        )
                        global_idx += 1
                    finally:
                        for var_name in (
                            "single_logits",
                            "single_inputs",
                            "single_input_values",
                            "single_attention_mask",
                        ):
                            locals().pop(var_name, None)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Free raw speech memory from batch dicts
                for item in batch:
                    item.pop("speech", None)
                locals().pop("speech_list", None)

                continue
            elif isinstance(e, NotImplementedError) and device == "mps":
                print(
                    "  MPS execution failed. Falling back to CPU backend for this batch...",
                    flush=True,
                )
                device = "cpu"
                model.to(device)  # type: ignore
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

        # Extract and unpad logits for each item
        for item_idx, item in enumerate(batch):
            input_len = len(item["speech"])
            logit_len = int(model._get_feat_extract_output_lengths(input_len))
            logits_np = batch_logits[item_idx, :logit_len].detach().cpu().numpy().copy()
            item_data = (global_idx, item["filename"], item["audio_path"], logits_np)
            max_queue.acquire()
            pool.apply_async(
                decode_worker, (item_data,), callback=write_result_callback
            )
            global_idx += 1

        pbar.update(len(batch))

        # Free batch tensors to ensure we don't peak VRAM on next batch allocation
        if "batch_logits" in locals():
            del batch_logits
        if "inputs" in locals():
            del inputs
        if "input_values" in locals():
            del input_values
        if "attention_mask" in locals():
            del attention_mask

        # Free raw speech memory from batch dicts
        for item in batch:
            item.pop("speech", None)
        if "speech_list" in locals():
            del speech_list

        # Keep VRAM heavily defragmented between every batch to accommodate long audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    pool.close()
    pool.join()

    print(
        f"Inference and decoding completed in {time.time() - gpu_start_time:.2f}s.",
        flush=True,
    )

    # Final sort of CSV back to original directory reading order
    print("Sorting final CSV output...", flush=True)
    wav_order = {path: idx for idx, path in enumerate(wav_files)}

    final_rows = []
    with open(args.output, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            final_rows.append(row)

    final_rows.sort(key=lambda x: wav_order.get(x[0], 0))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(final_rows)

    total_time = time.time() - audio_load_start
    print(
        f"\nBatch processing complete in {total_time:.2f}s. Results saved to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
