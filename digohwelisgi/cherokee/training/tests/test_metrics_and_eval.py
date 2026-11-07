# -*- coding: utf-8 -*-
"""
Unit tests for dual-head evaluation metrics and 50/50 interleaved dataset sampling (Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset, interleave_datasets
import numpy as np
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from digohwelisgi.cherokee.orthography.tones import (
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
    encode_cv_text,
    encode_tone_tokens,
)
from digohwelisgi.cherokee.training.train import compute_dual_metrics


def _build_processor(tmp_path: Path) -> Wav2Vec2Processor:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(SEGMENTAL_CV_VOCAB), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _one_hot_logits(seq_ids: list[int], vocab_size: int, seq_len: int) -> np.ndarray:
    pad_id = vocab_size - 1
    arr = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
    arr[:, pad_id] = 10.0
    # Space out tokens by 2 frames so CTC greedy collapse never merges identical adjacent tokens
    for idx, tok_id in enumerate(seq_ids):
        frame = idx * 2
        if frame < seq_len:
            arr[frame, :] = -10.0
            arr[frame, tok_id] = 10.0
    return arr


def test_compute_dual_metrics_exact_and_masked(tmp_path: Path) -> None:
    processor = _build_processor(tmp_path)

    cv_refs = [
        "na anitshutsa",
        "skwalhehsti",
        "atalenihskv",
        "tsisa kalonetv",
    ]
    tone_refs = [
        "C 3_1 | 2_1 C 2_2 C 3_2 C 2_1",
        "C C 2_2 C 3_2 C C 2_1",
        "",  # Bible unannotated
        "",  # Bible unannotated
    ]

    cv_id_seqs = [encode_cv_text(s) for s in cv_refs]
    tone_id_seqs = [encode_tone_tokens(t) for t in tone_refs]

    max_cv_len = max(len(x) for x in cv_id_seqs)
    max_tone_len = max(len(x) for x in tone_id_seqs)
    seq_len = max(max_cv_len, max_tone_len) * 2 + 4

    labels_cv = np.full((4, max_cv_len), -100, dtype=np.int64)
    labels_tone = np.full((4, max_tone_len), -100, dtype=np.int64)

    logits_cv = np.zeros((4, seq_len, len(SEGMENTAL_CV_VOCAB)), dtype=np.float32)
    logits_tone = np.zeros((4, seq_len, len(MORAIC_TONE_VOCAB)), dtype=np.float32)

    for i in range(4):
        labels_cv[i, : len(cv_id_seqs[i])] = cv_id_seqs[i]
        logits_cv[i] = _one_hot_logits(cv_id_seqs[i], len(SEGMENTAL_CV_VOCAB), seq_len)

        labels_tone[i, : len(tone_id_seqs[i])] = tone_id_seqs[i]
        if tone_refs[i]:
            logits_tone[i] = _one_hot_logits(
                tone_id_seqs[i], len(MORAIC_TONE_VOCAB), seq_len
            )
        else:
            # Put garbage tone predictions on Bible rows to prove they are strictly masked out!
            logits_tone[i] = _one_hot_logits(
                [MORAIC_TONE_VOCAB["4_2"]] * 5, len(MORAIC_TONE_VOCAB), seq_len
            )

    pred = SimpleNamespace(
        predictions=(logits_cv, logits_tone),
        label_ids=(labels_cv, labels_tone),
    )

    metrics = compute_dual_metrics(pred, processor, vocab_tone=MORAIC_TONE_VOCAB)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["tone_cer"] == 0.0
    assert metrics["reconciled_cer"] == 0.0

    # Now perturb only the first sample's tone prediction and verify tone_cer > 0 while cv cer == 0
    perturbed_tone_seq = list(tone_id_seqs[0])
    perturbed_tone_seq[1] = MORAIC_TONE_VOCAB["2_1"]  # change 3_1 -> 2_1
    logits_tone[0] = _one_hot_logits(
        perturbed_tone_seq, len(MORAIC_TONE_VOCAB), seq_len
    )

    pred_perturbed = SimpleNamespace(
        predictions=(logits_cv, logits_tone),
        label_ids=(labels_cv, labels_tone),
    )
    metrics_perturbed = compute_dual_metrics(
        pred_perturbed, processor, vocab_tone=MORAIC_TONE_VOCAB
    )
    assert metrics_perturbed["cer"] == 0.0
    assert metrics_perturbed["tone_cer"] > 0.0
    assert metrics_perturbed["reconciled_cer"] > 0.0


def test_interleaved_sampler() -> None:
    ds_orig = Dataset.from_dict(
        {
            "input_length": [16000] * 10,
            "labels": [encode_cv_text("na")] * 10,
            "tone_labels": [encode_tone_tokens("C 3_1")] * 10,
            "domain": ["feeling"] * 10,
        }
    )
    ds_bible = Dataset.from_dict(
        {
            "input_length": [16000] * 10,
            "labels": [encode_cv_text("tsisa")] * 10,
            "tone_labels": [encode_tone_tokens("")] * 10,
            "domain": ["bible"] * 10,
        }
    )

    interleaved = interleave_datasets(
        [ds_orig, ds_bible],
        probabilities=[0.5, 0.5],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    domains = [row["domain"] for row in interleaved]  # type: ignore[index]
    assert "feeling" in domains
    assert "bible" in domains
    for row in interleaved:
        r = dict(row)  # type: ignore[arg-type]
        if r["domain"] == "feeling":
            assert any(t != -100 for t in r["tone_labels"])
        else:
            assert r["tone_labels"] == [-100]
