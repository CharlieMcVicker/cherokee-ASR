#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_revisions.py

Checks out each model revision from revisions_to_test.tsv from Hugging Face Hub,
runs evaluation on the test dataset, and saves a scoring CSV.
"""

import os
import argparse
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset, Audio, Features, Value
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

# Import helper functions from evaluate_checkpoint and infer
from transcription.training.evaluate_checkpoint import (
    _try_read_csv,
    _detect_columns,
    _resolve_audio_path
)
from transcription.inference.infer import (
    greedy_inference,
    normalize_text,
    strip_tones,
    strip_length,
    TARGET_SAMPLE_RATE
)

def safe(s):
    return s if s.strip() else " "

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple model revisions on a test dataset.")
    parser.add_argument("--revisions-csv", type=str, default="data/results/revisions_to_test.tsv", help="Path to TSV or CSV listing revisions.")
    parser.add_argument("--test-csv", type=str, default="data/processed/cim-wav2vec2-test.csv", help="Path to the test CSV file.")
    parser.add_argument("--audio-dir", type=str, default="data/processed/sentence_audio", help="Directory containing audio files.")
    parser.add_argument("--checkpoint", type=str, default="charliemcvicker/length-only-20260702-173608-asr-cherokee", help="Hugging Face repo ID to the model checkpoint.")
    parser.add_argument("--output-csv", type=str, default="data/results/revision_scores.csv", help="Output path for the scoring CSV.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub authentication token.")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Loading revisions file: {args.revisions_csv}")
    revisions_list = []
    with open(args.revisions_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    start_idx = 0
    if lines and ("name" in lines[0].lower() or "revision" in lines[0].lower()):
        start_idx = 1
        
    for line in lines[start_idx:]:
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            name, rev_hash = parts[0].strip(), parts[1].strip()
            if name.endswith(","):
                name = name[:-1].strip()
            revisions_list.append((name, rev_hash, rev_hash))
        else:
            print(f"Skipping malformed line: {line}")

    print(f"Found {len(revisions_list)} revisions to evaluate.")

    print(f"Loading test CSV: {args.test_csv}")
    df_test = _try_read_csv(args.test_csv)
    audio_col, text_col = _detect_columns(df_test)
    print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")
    
    df_test[audio_col] = df_test[audio_col].apply(lambda p: _resolve_audio_path(p, args.audio_dir))
    df_test[text_col]  = df_test[text_col].apply(normalize_text)
    df_test.dropna(subset=[audio_col, text_col], inplace=True)
    print(f"Number of test items: {len(df_test)}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Preparing HuggingFace dataset...")
    data_dict = {
        "audio": df_test[audio_col].tolist(),
        "sentence": df_test[text_col].tolist()
    }
    features = Features({
        "audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
        "sentence": Value("string")
    })
    test_ds = Dataset.from_dict(data_dict, features=features)

    print(f"Loading initial processor from {args.checkpoint}...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint, token=token)

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        return batch
        
    test_ds_prepared = test_ds.map(prepare_batch, remove_columns=[c for c in test_ds.column_names if c != "sentence"], num_proc=1)

    scores = []

    for idx, (name, rev_hash, orig_rev_str) in enumerate(revisions_list):
        print(f"\n[{idx+1}/{len(revisions_list)}] Evaluating revision: {rev_hash} ({name})")
        
        try:
            model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint, token=token, revision=rev_hash)
            model.eval()
            model.to(device)
            
            try:
                current_processor = Wav2Vec2Processor.from_pretrained(args.checkpoint, token=token, revision=rev_hash)
            except Exception:
                current_processor = processor

            results = []
            
            for ex in test_ds_prepared:
                input_values = torch.tensor([ex["input_values"]]).to(device)
                with torch.no_grad():
                    logits = model(input_values=input_values).logits
                
                sliced_logits = logits[0]
                res = greedy_inference(sliced_logits, current_processor)
                hyp_greedy = res["text"]
                
                gold = ex["sentence"]
                
                row_res = {
                    "gold": gold,
                    "greedy": hyp_greedy,
                }
                results.append(row_res)
            
            # Calculate overall metrics
            golds = [r["gold"] for r in results]
            greedies = [r["greedy"] for r in results]
            
            wer_greedy = jiwer_wer(golds, greedies)
            cer_greedy = jiwer_cer(golds, greedies)
            
            golds_masked = [strip_length(g) for g in golds]
            greedies_masked = [strip_length(g) for g in greedies]
            
            wer_greedy_masked = jiwer_wer(golds_masked, greedies_masked)
            cer_greedy_masked = jiwer_cer(golds_masked, greedies_masked)
            
            score_entry = {
                "name": name,
                "revision": rev_hash,
                "greedy_wer": wer_greedy,
                "greedy_cer": cer_greedy,
                "greedy_masked_wer": wer_greedy_masked,
                "greedy_masked_cer": cer_greedy_masked,
            }
            
            print(f"  Greedy: Standard WER = {wer_greedy:.4f} (CER = {cer_greedy:.4f}) | Masked WER = {wer_greedy_masked:.4f} (CER = {cer_greedy_masked:.4f})")
            scores.append(score_entry)
            
        except Exception as e:
            print(f"Error evaluating revision {rev_hash}: {e}")
            
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully saved scoring results to {args.output_csv}")

if __name__ == "__main__":
    main()
