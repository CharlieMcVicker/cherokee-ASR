import os
import re
import glob
from typing import Any
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transcription.models.asr_model import CherokeeASRModel

from jiwer import wer as jiwer_wer, cer as jiwer_cer


from transcription.inference.infer import (
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
    Yields (label, asr_model, processor, path) for all checkpoints in checkpoints_dir.
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

    for label, path in checkpoints:
        asr_model = CherokeeASRModel.from_pretrained(
            path_or_repo=path,
            processor_path=processor_path,
            device="cpu",
            use_cache=False,
        )
        yield label, asr_model, asr_model.processor, path
        del asr_model


def yield_hf_revisions(repo_id, revisions, token=None):
    """
    Yields (label, asr_model, processor, path) for each revision in revisions list.
    """
    for friendly_name, rev_hash in revisions:
        try:
            asr_model = CherokeeASRModel.from_pretrained(
                path_or_repo=repo_id,
                revision=rev_hash,
                token=token,
                device="cpu",
                use_cache=False,
            )
        except Exception:
            asr_model = CherokeeASRModel.from_pretrained(
                path_or_repo=repo_id,
                revision=rev_hash,
                processor_path=repo_id,
                token=token,
                device="cpu",
                use_cache=False,
            )
        yield f"{friendly_name} ({rev_hash[:7]})", asr_model, asr_model.processor, f"hf://{repo_id}@{rev_hash}"
        del asr_model


def yield_single_checkpoint(
    checkpoint_path: str,
    processor_path: str | None = None,
    revision: str | None = None,
    token: str | None = None,
):
    """
    Yields a single local or HF checkpoint/model as CherokeeASRModel.
    """
    proc_path = processor_path or checkpoint_path
    asr_model = CherokeeASRModel.from_pretrained(
        path_or_repo=checkpoint_path,
        revision=revision,
        processor_path=proc_path,
        token=token,
        device="cpu",
        use_cache=False,
    )

    yield "checkpoint", asr_model, asr_model.processor, checkpoint_path
    del asr_model


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

    for item in model_generator:
        if len(item) == 4:
            label, asr_model_or_model, processor, path = item
        else:
            label, asr_model_or_model, path = item
            processor = None

        print(f"Evaluating model: {label} ({path})")
        if isinstance(asr_model_or_model, CherokeeASRModel):
            asr_model = asr_model_or_model
            asr_model.to(device)
        else:
            asr_model = CherokeeASRModel(asr_model_or_model, processor, device=device)

        asr_model.model.eval()

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

            curr_collator = SimpleDataCollator(asr_model.processor)
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
            input_values = batch["input_values"].to(asr_model.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(asr_model.device)

            with torch.no_grad():
                outputs = asr_model.model(
                    input_values=input_values, attention_mask=attention_mask
                )
                logits = outputs.logits

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1)
                feat_extractor = getattr(
                    asr_model.model, "_get_feat_extract_output_lengths", None
                )
                if callable(feat_extractor):
                    raw_lens: Any = feat_extractor(input_lengths)
                    output_lengths = (
                        raw_lens.cpu().numpy()
                        if hasattr(raw_lens, "cpu")
                        else np.array(raw_lens)
                    )
                else:
                    output_lengths = [logits.shape[1]] * logits.shape[0]
            else:
                output_lengths = [logits.shape[1]] * logits.shape[0]

            logits_np = logits.cpu().numpy()
            for i in range(len(logits_np)):
                all_logits.append(logits_np[i, : int(output_lengths[i]), :])

        all_gold = [ex["sentence"] for ex in test_ds_prepared]

        transforms = [
            ("raw", lambda x: x),
            ("len_masked", strip_length),
            ("tone_masked", strip_tones),
            ("both_masked", strip_both),
        ]

        ckpt_rows = []
        for idx, logits in enumerate(all_logits):
            ref = all_gold[idx]
            res = asr_model.decode(logits, compute_word_confidences=False)
            hyp = res.text

            row = {
                "checkpoint": label,
                "index": idx,
                "gold": ref,
                "hyp_greedy": hyp,
            }
            for name, transform_fn in transforms:
                ref_t = safe(transform_fn(ref))
                hyp_t = safe(transform_fn(hyp))
                wer_val = jiwer_wer(ref_t, hyp_t)
                cer_val = jiwer_cer(ref_t, hyp_t)
                row[f"wer_{name}"] = wer_val
                row[f"cer_{name}"] = cer_val
                if name == "raw":
                    row["wer_greedy"] = wer_val
                    row["cer_greedy"] = cer_val
                elif name == "len_masked":
                    row["wer_greedy_masked"] = wer_val
                    row["cer_greedy_masked"] = cer_val

            ckpt_rows.append(row)

        rows_by_ckpt[label] = ckpt_rows

        df = pd.DataFrame(ckpt_rows)
        greedies = list(df["hyp_greedy"])

        agg_metrics = {}
        for name, transform_fn in transforms:
            ref_list = [safe(transform_fn(g)) for g in all_gold]
            hyp_list = [safe(transform_fn(h)) for h in greedies]
            agg_metrics[f"agg_wer_{name}"] = jiwer_wer(ref_list, hyp_list)
            agg_metrics[f"agg_cer_{name}"] = jiwer_cer(ref_list, hyp_list)

        ranking.append(
            {
                "checkpoint": label,
                "path": path,
                "median_wer_greedy": float(np.median(df["wer_greedy"])),
                **agg_metrics,
                "agg_wer_greedy": agg_metrics["agg_wer_raw"],
                "agg_cer_greedy": agg_metrics["agg_cer_raw"],
                "agg_wer_length_masked": agg_metrics["agg_wer_len_masked"],
                "agg_cer_length_masked": agg_metrics["agg_cer_len_masked"],
                "agg_wer_greedy_masked": agg_metrics["agg_wer_len_masked"],
                "agg_cer_greedy_masked": agg_metrics["agg_cer_len_masked"],
            }
        )

        clean_eval_cache(device)
        asr_model.to("cpu")
        del asr_model
        clean_eval_cache(device)

    ranking_df = pd.DataFrame(ranking)
    return rows_by_ckpt, ranking_df
