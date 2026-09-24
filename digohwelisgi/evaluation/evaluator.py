# -*- coding: utf-8 -*-
"""
digohwelisgi.evaluation.evaluator

End-to-end multi-SNR evaluation engine, CTC emission peak and top-K posterior
extraction, and streaming JSONL checkpoint resume support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel
import torch

try:
    from jiwer import cer as jiwer_cer
except ImportError:
    jiwer_cer = None

from digohwelisgi.evaluation.confusion import character_levenshtein_align
from digohwelisgi.evaluation.perturbations import AudioTransform
from digohwelisgi.cherokee.orthography import normalize_text
from digohwelisgi.cherokee.models import CherokeeASRModel

logger = logging.getLogger(__name__)


class EvaluationRecord(BaseModel):
    """
    Structured record representing the evaluation of a single audio sample
    under a specific noise condition and SNR tier.
    """

    audio_id: str
    reference: str
    snr_tier: float
    noise_type: str
    hypothesis: str
    top_k_tokens: list[list[tuple[str, float]]]
    cer: float
    metadata: Optional[dict[str, Any]] = None


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using jiwer if available, or Levenshtein distance fallback.
    """
    ref_safe = str(reference).strip() if reference is not None else ""
    hyp_safe = str(hypothesis).strip() if hypothesis is not None else ""
    if not ref_safe and not hyp_safe:
        return 0.0
    if not ref_safe:
        return float(len(hyp_safe))
    if jiwer_cer is not None:
        return float(jiwer_cer(ref_safe, hyp_safe if hyp_safe else " "))
    pairs = character_levenshtein_align(ref_safe, hyp_safe)
    errors = sum(1 for _, _, op in pairs if op != "match")
    return float(errors / max(1, len(ref_safe)))


class NoisyEvaluator:
    """
    Evaluator that applies acoustic perturbations, extracts CTC non-blank emissions
    and top-K candidate token posteriors, calculates CER, and streams evaluation
    records to JSONL with checkpoint resume support.
    """

    def __init__(
        self,
        model: Optional[Union[CherokeeASRModel, Any]] = None,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        top_k: int = 5,
    ) -> None:
        self.top_k = int(top_k)
        self.device = str(device) if device is not None else None
        if model is not None:
            self.model = model
            if self.device is not None and hasattr(self.model, "to"):
                self.model.to(self.device)
        else:
            self.model = CherokeeASRModel.from_pretrained_or_best(
                path_or_repo=model_path,
                device=self.device,
            )
        self.processor = getattr(self.model, "processor", None)

    def extract_peaks_and_topk(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        processor: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> tuple[str, list[list[tuple[str, float]]]]:
        """
        Given frame-level softmax probabilities [T, V], extract CTC emission sequence
        and top-K alternative token candidates per non-blank emission position.
        """
        proc = processor or self.processor
        if proc is None:
            raise ValueError("A valid processor is required to extract tokens.")

        if isinstance(probs, torch.Tensor):
            probs_arr = probs.detach().cpu().numpy()
        else:
            probs_arr = np.asarray(probs)

        if probs_arr.ndim == 3:
            probs_arr = probs_arr.squeeze(0)

        k_val = top_k if top_k is not None else self.top_k
        k_val = min(k_val, probs_arr.shape[-1])

        pred_ids = np.argmax(probs_arr, axis=-1)

        pad_id = getattr(proc.tokenizer, "pad_token_id", None)
        if (
            pad_id is None
            and hasattr(proc, "tokenizer")
            and hasattr(proc.tokenizer, "vocab")
        ):
            pad_id = proc.tokenizer.vocab.get("[PAD]", 0)
        if pad_id is None:
            pad_id = 0

        word_delim = getattr(proc.tokenizer, "word_delimiter_token_id", None)
        if (
            word_delim is None
            and hasattr(proc, "tokenizer")
            and hasattr(proc.tokenizer, "vocab")
        ):
            word_delim = proc.tokenizer.vocab.get("|", None)

        chars: list[str] = []
        top_k_list: list[list[tuple[str, float]]] = []
        top_k_indices = np.argsort(probs_arr, axis=-1)[:, -k_val:][:, ::-1]
        top_k_probs = np.take_along_axis(probs_arr, top_k_indices, axis=-1)

        prev_id = -1
        for t, token_id in enumerate(pred_ids):
            if token_id != pad_id and token_id != prev_id:
                if word_delim is not None and token_id == word_delim:
                    token_str = " "
                else:
                    token_str = proc.decode([token_id])

                if token_str:
                    chars.append(token_str)
                    alts: list[tuple[str, float]] = []
                    for k in range(k_val):
                        alt_id = int(top_k_indices[t, k])
                        alt_prob = float(top_k_probs[t, k])
                        if word_delim is not None and alt_id == word_delim:
                            alt_str = " "
                        else:
                            alt_str = proc.decode([alt_id])
                        if alt_str:
                            alts.append((alt_str, alt_prob))
                    top_k_list.append(alts)
            prev_id = token_id

        return "".join(chars), top_k_list

    def evaluate_waveform(
        self,
        waveform: Union[torch.Tensor, np.ndarray, str, bytes, List[float]],
        reference: str,
        audio_id: str,
        sample_rate: int = 16000,
        snr_tier: float = 25.0,
        noise_type: str = "white",
        perturbation: Optional[AudioTransform] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EvaluationRecord:
        """
        Applies perturbation if given, passes audio through model.get_probabilities,
        extracts hypothesis and top_k, calculates CER, and returns an EvaluationRecord.
        """
        # Ensure waveform is torch.Tensor
        if isinstance(waveform, torch.Tensor):
            wav_tensor = waveform.clone().detach().to(torch.float32)
        elif isinstance(waveform, np.ndarray):
            wav_tensor = torch.tensor(waveform, dtype=torch.float32)
        elif isinstance(waveform, (str, bytes, list)):
            wav_arr = CherokeeASRModel.preprocess_audio(
                waveform, sample_rate=sample_rate
            )
            wav_tensor = torch.tensor(wav_arr, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported waveform input type: {type(waveform)}")

        # Apply perturbation if provided
        if perturbation is not None:
            wav_tensor = perturbation(wav_tensor, sample_rate)

        # Get probabilities from model
        with torch.no_grad():
            probs = self.model.get_probabilities(wav_tensor, sample_rate=sample_rate)
            if isinstance(probs, torch.Tensor):
                probs_arr = probs.detach().cpu().numpy()
            else:
                probs_arr = np.asarray(probs)

        if probs_arr.ndim == 3:
            probs_arr = probs_arr.squeeze(0)

        # Extract CTC peak tokens and top_k alternatives
        hyp, top_k_tokens = self.extract_peaks_and_topk(
            probs_arr, processor=self.processor
        )

        # Normalize reference according to training loop standard (removes colons, punctuation)
        clean_ref = normalize_text(reference)

        # Calculate CER
        cer_score = calculate_cer(clean_ref, hyp)

        return EvaluationRecord(
            audio_id=str(audio_id),
            reference=clean_ref,
            snr_tier=float(snr_tier),
            noise_type=str(noise_type),
            hypothesis=hyp,
            top_k_tokens=top_k_tokens,
            cer=cer_score,
            metadata=metadata,
        )

    def stream_evaluate_dataset(
        self,
        samples: Sequence[dict[str, Any]],
        perturbations_by_tier: Sequence[Tuple[float, str, Optional[AudioTransform]]],
        output_jsonl_path: Union[str, Path],
        resume: bool = True,
        sample_rate: int = 16000,
    ) -> list[EvaluationRecord]:
        """
        Stream evaluate a collection of dataset samples across perturbation tiers.
        Writes completed EvaluationRecords directly to JSONL with checkpoint resume support.
        """
        out_path = Path(output_jsonl_path)
        existing_keys: set[tuple[str, float, str]] = set()
        records: list[EvaluationRecord] = []

        if resume and out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        rec = EvaluationRecord.model_validate_json(line_str)
                        existing_keys.add(
                            (
                                str(rec.audio_id),
                                float(rec.snr_tier),
                                str(rec.noise_type),
                            )
                        )
                        records.append(rec)
                    except Exception as e:
                        logger.warning(
                            "Failed to parse existing JSONL record in %s: %s",
                            out_path,
                            e,
                        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        if not resume:
            records = []
            existing_keys = set()

        with open(out_path, mode, encoding="utf-8") as f:
            for idx, sample in enumerate(samples):
                audio_id = sample.get("audio_id") or sample.get("id")
                if audio_id is None:
                    if "path" in sample and sample["path"]:
                        audio_id = Path(sample["path"]).stem
                    else:
                        audio_id = f"sample_{idx}"
                audio_id = str(audio_id)

                reference = sample.get("reference")
                if reference is None:
                    reference = sample.get("sentence", "")
                reference = str(reference)

                metadata = sample.get("metadata")

                # Filter tiers that still need evaluation
                missing_tiers = [
                    (snr, ntype, tf)
                    for snr, ntype, tf in perturbations_by_tier
                    if not (
                        resume and (audio_id, float(snr), str(ntype)) in existing_keys
                    )
                ]
                if not missing_tiers:
                    continue

                # Preload waveform once per sample across all missing tiers
                waveform = sample.get("waveform")
                if waveform is None:
                    waveform = sample.get("audio")
                if waveform is None and "path" in sample and sample["path"]:
                    waveform = sample["path"]

                if isinstance(waveform, (str, bytes, list)):
                    try:
                        wav_arr = CherokeeASRModel.preprocess_audio(
                            waveform, sample_rate=sample_rate
                        )
                        waveform = torch.tensor(wav_arr, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(
                            "Failed to load audio for sample %s (%s): %s",
                            audio_id,
                            waveform,
                            e,
                        )
                        continue
                elif isinstance(waveform, np.ndarray):
                    waveform = torch.tensor(waveform, dtype=torch.float32)

                if waveform is None:
                    logger.warning(
                        "No waveform or audio path found for sample %s", audio_id
                    )
                    continue

                for snr_tier, noise_type, transform in missing_tiers:
                    key = (audio_id, float(snr_tier), str(noise_type))

                    rec = self.evaluate_waveform(
                        waveform=waveform,
                        reference=reference,
                        audio_id=audio_id,
                        sample_rate=sample_rate,
                        snr_tier=snr_tier,
                        noise_type=noise_type,
                        perturbation=transform,
                        metadata=metadata,
                    )
                    records.append(rec)
                    existing_keys.add(key)
                    f.write(rec.model_dump_json() + "\n")
                    f.flush()

                if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
                    logger.info(
                        "Evaluated %d/%d samples (%d records accumulated)...",
                        idx + 1,
                        len(samples),
                        len(records),
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        return records
