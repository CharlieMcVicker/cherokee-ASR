import os
import re
import glob
import torch
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer as jiwer_wer, cer as jiwer_cer

from transcription.inference.infer import (
    greedy_inference,
    strip_tones,
    strip_length,
    strip_both,
)


def get_eval_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clean_eval_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def yield_local_checkpoints(checkpoints_dir, processor_path=None):
    """
    Yields (label, model, processor, path) for all checkpoints in checkpoints_dir.
    """
    ckpt_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))

    def _step(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpt_dirs = sorted(ckpt_dirs, key=_step)

    checkpoints = []
    for d in ckpt_dirs:
        if os.path.exists(os.path.join(d, "config.json")):
            checkpoints.append((os.path.basename(d), d))

    if os.path.exists(os.path.join(checkpoints_dir, "config.json")):
        checkpoints.append(("final", checkpoints_dir))

    if not checkpoints:
        return

    # Find processor
    if not processor_path:
        for cand in [checkpoints_dir, checkpoints[0][1]]:
            if os.path.exists(os.path.join(cand, "vocab.json")):
                processor_path = cand
                break
        if not processor_path:
            processor_path = checkpoints_dir

    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    for label, path in checkpoints:
        model = Wav2Vec2ForCTC.from_pretrained(path)
        yield label, model, processor, path
        del model


def yield_hf_revisions(repo_id, revisions, token=None):
    """
    Yields (label, model, processor, path) for each revision in revisions list.
    """
    fallback_processor = Wav2Vec2Processor.from_pretrained(repo_id, token=token)
    for friendly_name, rev_hash in revisions:
        model = Wav2Vec2ForCTC.from_pretrained(repo_id, token=token, revision=rev_hash)
        try:
            processor = Wav2Vec2Processor.from_pretrained(
                repo_id, token=token, revision=rev_hash
            )
        except Exception:
            processor = fallback_processor
        yield f"{friendly_name} ({rev_hash[:7]})", model, processor, f"hf://{repo_id}@{rev_hash}"
        del model


def yield_single_checkpoint(
    checkpoint_path, processor_path=None, revision=None, token=None
):
    """
    Yields a single local or HF checkpoint/model.
    """
    proc_path = processor_path or checkpoint_path
    processor = Wav2Vec2Processor.from_pretrained(
        proc_path, token=token, revision=revision
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        checkpoint_path, token=token, revision=revision
    )
    yield "checkpoint", model, processor, checkpoint_path
    del model


def run_evaluation(
    model_generator, test_ds_prepared, data_collator=None, batch_size=16
):
    """
    Runs the full evaluation pipeline, computing unmasked, vowel-length-masked,
    tone-masked, and both-masked metrics.
    """
    device = get_eval_device()

    def safe(s):
        return s if s.strip() else " "

    rows_by_ckpt = {}
    ranking = []

    for label, model, processor, path in model_generator:
        print(f"Evaluating model: {label} ({path})")
        model.eval()
        model.to(device)

        # If no data_collator provided, create one dynamically
        if data_collator is None:

            class SimpleDataCollator:
                def __init__(self, proc):
                    self.proc = proc

                def __call__(self, features):
                    input_features = [
                        {"input_values": f["input_values"]} for f in features
                    ]
                    return self.proc.pad(
                        input_features, padding=True, return_tensors="pt"
                    )

            curr_collator = SimpleDataCollator(processor)
        else:
            curr_collator = data_collator

        test_loader = DataLoader(
            test_ds_prepared,
            batch_size=batch_size,
            collate_fn=curr_collator,
            shuffle=False,
        )

        all_logits = []

        # We run inference in batches
        for batch in tqdm(test_loader, desc=f"Inference ({label})"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(
                    input_values=input_values, attention_mask=attention_mask
                )
                logits = outputs.logits

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1)
                output_lengths = (
                    model._get_feat_extract_output_lengths(input_lengths).cpu().numpy()
                )
            else:
                output_lengths = [logits.shape[1]] * logits.shape[0]

            logits_np = logits.cpu().numpy()
            for i in range(len(logits_np)):
                all_logits.append(logits_np[i, : int(output_lengths[i]), :])

        all_gold = [ex["sentence"] for ex in test_ds_prepared]

        ckpt_rows = []
        for idx, logits in enumerate(all_logits):
            ref = all_gold[idx]
            res = greedy_inference(logits, processor)
            hyp = res["text"]

            ref_safe, hyp_safe = safe(ref), safe(hyp)

            # Vowel length masked (used in train.py / local eval)
            ref_len_masked = safe(strip_length(ref))
            hyp_len_masked = safe(strip_length(hyp))

            # Tone masked
            ref_tone_masked = safe(strip_tones(ref))
            hyp_tone_masked = safe(strip_tones(hyp))

            # Both masked
            ref_both_masked = safe(strip_both(ref))
            hyp_both_masked = safe(strip_both(hyp))

            ckpt_rows.append(
                {
                    "checkpoint": label,
                    "index": idx,
                    "gold": ref,
                    "hyp_greedy": hyp,
                    "wer_greedy": jiwer_wer(ref_safe, hyp_safe),
                    "cer_greedy": jiwer_cer(ref_safe, hyp_safe),
                    "wer_greedy_masked": jiwer_wer(ref_len_masked, hyp_len_masked),
                    "cer_greedy_masked": jiwer_cer(ref_len_masked, hyp_len_masked),
                    "wer_tone_masked": jiwer_wer(ref_tone_masked, hyp_tone_masked),
                    "cer_tone_masked": jiwer_cer(ref_tone_masked, hyp_tone_masked),
                    "wer_both_masked": jiwer_wer(ref_both_masked, hyp_both_masked),
                    "cer_both_masked": jiwer_cer(ref_both_masked, hyp_both_masked),
                }
            )

        rows_by_ckpt[label] = ckpt_rows

        df = pd.DataFrame(ckpt_rows)
        golds = list(df["gold"])
        greedies = list(df["hyp_greedy"])

        ranking.append(
            {
                "checkpoint": label,
                "path": path,
                "median_wer_greedy": float(np.median(df["wer_greedy"])),
                "median_cer_greedy": float(np.median(df["cer_greedy"])),
                "agg_wer_greedy": jiwer_wer(golds, greedies),
                "agg_cer_greedy": jiwer_cer(golds, greedies),
                "median_wer_greedy_masked": float(np.median(df["wer_greedy_masked"])),
                "median_cer_greedy_masked": float(np.median(df["cer_greedy_masked"])),
                "agg_wer_greedy_masked": jiwer_wer(
                    [safe(strip_length(g)) for g in golds],
                    [safe(strip_length(h)) for h in greedies],
                ),
                "agg_cer_greedy_masked": jiwer_cer(
                    [safe(strip_length(g)) for g in golds],
                    [safe(strip_length(h)) for h in greedies],
                ),
                "agg_wer_tone_masked": jiwer_wer(
                    [safe(strip_tones(g)) for g in golds],
                    [safe(strip_tones(h)) for h in greedies],
                ),
                "agg_cer_tone_masked": jiwer_cer(
                    [safe(strip_tones(g)) for g in golds],
                    [safe(strip_tones(h)) for h in greedies],
                ),
                "agg_wer_both_masked": jiwer_wer(
                    [safe(strip_both(g)) for g in golds],
                    [safe(strip_both(h)) for h in greedies],
                ),
                "agg_cer_both_masked": jiwer_cer(
                    [safe(strip_both(g)) for g in golds],
                    [safe(strip_both(h)) for h in greedies],
                ),
            }
        )

        clean_eval_cache(device)

    ranking_df = pd.DataFrame(ranking)
    return rows_by_ckpt, ranking_df
