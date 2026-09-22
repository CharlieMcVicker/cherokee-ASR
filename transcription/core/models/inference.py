# -*- coding: utf-8 -*-
"""
inference.py

Standalone procedural inference pipeline for extracting CTC emissions from raw model/processor.
Operates on raw Hugging Face Wav2Vec2 models and processors with optional on-disk .npz caching.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from transcription.core.models.output import ModelOutput

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
FRAME_DURATION_SEC = 0.02


def _resolve_device(
    device: Optional[Union[str, torch.device]], model: Any
) -> Union[str, torch.device]:
    """Resolve target torch device safely handling mocks."""
    if device is not None and (isinstance(device, (str, torch.device))):
        return device
    model_dev = getattr(model, "device", None)
    if model_dev is not None and isinstance(model_dev, (str, torch.device)):
        return model_dev
    return "cpu"


def preprocess_audio(
    audio_input: Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor],
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Normalize and resample audio input to 1D float32 numpy array at 16kHz.

    Supports:
        - File path (str or Path)
        - Raw PCM bytes (float32)
        - Sequence of floats / Python list
        - NumPy ndarray (1D or 2D stereo averaged to mono)
        - PyTorch Tensor (1D or 2D stereo averaged to mono)
    """
    if isinstance(audio_input, (str, Path)):
        path_str = str(audio_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Audio file '{path_str}' not found.")
        speech_array, sr = sf.read(path_str)
        waveform = torch.tensor(speech_array, dtype=torch.float32)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy().astype(np.float32)

    elif isinstance(audio_input, bytes):
        speech = np.frombuffer(audio_input, dtype=np.float32)
    elif isinstance(audio_input, (list, tuple)):
        speech = np.array(audio_input, dtype=np.float32)
    elif isinstance(audio_input, torch.Tensor):
        speech = audio_input.detach().cpu().numpy().astype(np.float32)
    elif isinstance(audio_input, np.ndarray):
        speech = audio_input.astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=-1)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        speech = resampler(waveform).squeeze(0).numpy().astype(np.float32)

    return speech


def extract_vocab_and_metadata(processor: Any) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Extracts vocabulary mapping and tokenizer metadata (pad_token_id, word_delimiter_token_id)
    from a Wav2Vec2Processor.
    """
    vocab: dict[str, int] = {}
    metadata: dict[str, Any] = {}

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        raw_vocab = None
        if hasattr(tokenizer, "get_vocab") and callable(tokenizer.get_vocab):
            raw_vocab = tokenizer.get_vocab()
        if not isinstance(raw_vocab, (dict, Mapping)) and hasattr(tokenizer, "vocab"):
            raw_vocab = tokenizer.vocab

        if isinstance(raw_vocab, (dict, Mapping)):
            vocab = {str(k): int(v) for k, v in raw_vocab.items()}

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if (pad_id is None or not isinstance(pad_id, (int, np.integer))) and vocab:
            pad_id = vocab.get("[PAD]", 0)
        metadata["pad_token_id"] = (
            int(pad_id) if isinstance(pad_id, (int, np.integer)) else 0
        )

        delim_id = getattr(tokenizer, "word_delimiter_token_id", None)
        metadata["word_delimiter_token_id"] = (
            int(delim_id) if isinstance(delim_id, (int, np.integer)) else None
        )
    elif hasattr(processor, "vocab") and isinstance(processor.vocab, (dict, Mapping)):
        vocab = {str(k): int(v) for k, v in processor.vocab.items()}

    return vocab, metadata


def compute_audio_cache_key(
    audio_input: Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor],
    model_identifier: str = "asr_model",
) -> str:
    """
    Generates a deterministic cache key for an audio input and model identity.
    """
    hasher = hashlib.sha256()
    hasher.update(model_identifier.encode("utf-8"))

    if isinstance(audio_input, (str, Path)):
        p = Path(audio_input)
        if p.exists() and p.is_file():
            try:
                st = p.stat()
                raw_id = f"|file:{p.resolve()}:{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                raw_id = f"|file:{p.resolve()}"
        else:
            raw_id = f"|path:{str(audio_input)}"
        hasher.update(raw_id.encode("utf-8"))
        stem = p.stem
        return f"{stem}_{hasher.hexdigest()[:16]}"

    stem = "audio"
    if isinstance(audio_input, bytes):
        hasher.update(b"|bytes|")
        hasher.update(audio_input)
    elif isinstance(audio_input, np.ndarray):
        hasher.update(
            f"|ndarray:{audio_input.shape}:{audio_input.dtype}|".encode("utf-8")
        )
        hasher.update(audio_input.tobytes())
    elif isinstance(audio_input, torch.Tensor):
        arr = audio_input.detach().cpu().numpy()
        hasher.update(f"|tensor:{arr.shape}:{arr.dtype}|".encode("utf-8"))
        hasher.update(arr.tobytes())
    else:
        hasher.update(f"|seq:{repr(audio_input)}|".encode("utf-8"))

    return f"{stem}_{hasher.hexdigest()[:16]}"


def infer_emissions(
    model: Any,
    processor: Any,
    audio_input: Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor],
    sample_rate: int = TARGET_SAMPLE_RATE,
    device: Optional[Union[str, torch.device]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    model_identifier: Optional[str] = None,
) -> ModelOutput:
    """
    Standalone procedure: Executes forward inference for single audio input and returns ModelOutput.
    If cache_dir is specified, checks for and writes to an on-disk .npz cache file.

    Args:
        model: Wav2Vec2ForCTC or compatible model.
        processor: Wav2Vec2Processor or compatible processor.
        audio_input: File path, PCM array, tensor, or bytes.
        sample_rate: Input sample rate (default: 16000).
        device: PyTorch device ('cpu', 'cuda', 'mps', or torch.device).
        cache_dir: Optional directory to cache and retrieve ModelOutput .npz files.
        model_identifier: Optional identifier for cache key generation.

    Returns:
        ModelOutput containing log probabilities (lpz), vocabulary mapping, and decoding methods.
    """
    resolved_model_id = model_identifier or getattr(model, "name_or_path", "model")
    cache_path: Optional[Path] = None

    if cache_dir is not None:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        key = compute_audio_cache_key(
            audio_input, model_identifier=str(resolved_model_id)
        )
        cache_path = c_dir / f"{key}.npz"
        if cache_path.exists():
            try:
                return ModelOutput.load(cache_path)
            except Exception as e:
                logger.warning(
                    "Failed to load cached ModelOutput from %s (%s). Recomputing...",
                    cache_path,
                    e,
                )

    speech = preprocess_audio(audio_input, sample_rate=sample_rate)

    inputs = processor(speech, sampling_rate=TARGET_SAMPLE_RATE)
    raw_input_values = inputs.input_values[0]
    input_tensor = torch.tensor(np.array([raw_input_values]), dtype=torch.float32)

    target_device = _resolve_device(device, model)
    input_tensor = input_tensor.to(target_device)

    try:
        with torch.no_grad():
            logits = model(input_tensor).logits
    except NotImplementedError as e:
        if str(target_device) == "mps":
            logger.warning(
                "MPS inference failed with NotImplementedError. Falling back to CPU."
            )
            fallback_device = "cpu"
            model.to(fallback_device)
            input_tensor = input_tensor.to(fallback_device)
            with torch.no_grad():
                logits = model(input_tensor).logits
        else:
            raise e

    if logits.ndim == 3:
        logits = logits.squeeze(0)

    lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
    vocab, metadata = extract_vocab_and_metadata(processor)
    metadata["sample_rate"] = TARGET_SAMPLE_RATE
    metadata["duration_sec"] = round(len(speech) / TARGET_SAMPLE_RATE, 4)

    output = ModelOutput(
        lpz=lpz,
        vocab=vocab,
        frame_duration_sec=FRAME_DURATION_SEC,
        metadata=metadata,
    )

    if cache_path is not None:
        try:
            output.save(cache_path)
        except Exception as e:
            logger.warning("Failed to save ModelOutput cache to %s (%s)", cache_path, e)

    return output


def infer_emissions_batch(
    model: Any,
    processor: Any,
    audio_inputs: Sequence[
        Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor]
    ],
    sample_rate: int = TARGET_SAMPLE_RATE,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 16,
    cache_dir: Optional[Union[str, Path]] = None,
    model_identifier: Optional[str] = None,
) -> List[ModelOutput]:
    """
    Standalone procedure: Batched inference for multiple audio inputs returning List[ModelOutput].
    Applies dynamic padding, handles OOM fallbacks, and checks/populates cache_dir if provided.

    Args:
        model: Wav2Vec2ForCTC or compatible model.
        processor: Wav2Vec2Processor or compatible processor.
        audio_inputs: Sequence of audio inputs (paths, PCM arrays, etc.).
        sample_rate: Input sample rate.
        device: Target execution device.
        batch_size: Batch size for GPU/CPU forward passes.
        cache_dir: Optional directory for .npz emission caching.
        model_identifier: Optional identifier for cache keys.

    Returns:
        List of ModelOutput instances matching the input order.
    """
    if not audio_inputs:
        return []

    resolved_model_id = model_identifier or getattr(model, "name_or_path", "model")
    target_device = _resolve_device(device, model)
    vocab, metadata = extract_vocab_and_metadata(processor)

    # First check caching for each item
    outputs: List[Optional[ModelOutput]] = [None] * len(audio_inputs)
    miss_indices: List[int] = []

    if cache_dir is not None:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(audio_inputs):
            key = compute_audio_cache_key(item, model_identifier=str(resolved_model_id))
            cache_path = c_dir / f"{key}.npz"
            if cache_path.exists():
                try:
                    outputs[idx] = ModelOutput.load(cache_path)
                    continue
                except Exception as e:
                    logger.warning(
                        "Failed to load cached ModelOutput for item %d (%s). Recomputing...",
                        idx,
                        e,
                    )
            miss_indices.append(idx)
    else:
        miss_indices = list(range(len(audio_inputs)))

    if not miss_indices:
        return [o for o in outputs if o is not None]

    # Process cache misses in batches
    for batch_start in range(0, len(miss_indices), batch_size):
        curr_miss_indices = miss_indices[batch_start : batch_start + batch_size]
        batch_speech: List[np.ndarray] = []

        for idx in curr_miss_indices:
            item = audio_inputs[idx]
            try:
                sp = preprocess_audio(item, sample_rate=sample_rate)
                batch_speech.append(sp)
            except Exception as e:
                logger.warning("Error preprocessing batch audio item %d: %s", idx, e)
                batch_speech.append(np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32))

        inputs = processor(
            batch_speech,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(target_device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        batch_logits: Optional[torch.Tensor] = None
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
            if is_oom or "cudnn" in err_str:
                logger.warning(
                    "OOM or cuDNN error in batch inference. Falling back to sequential."
                )
                del input_values
                if attention_mask is not None:
                    del attention_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                for sub_idx in curr_miss_indices:
                    sub_out = infer_emissions(
                        model=model,
                        processor=processor,
                        audio_input=audio_inputs[sub_idx],
                        sample_rate=TARGET_SAMPLE_RATE,
                        device=target_device,
                        cache_dir=cache_dir,
                        model_identifier=resolved_model_id,
                    )
                    outputs[sub_idx] = sub_out
                continue
            elif isinstance(e, NotImplementedError) and str(target_device) == "mps":
                logger.warning("MPS error in batch. Falling back to CPU.")
                target_device = "cpu"
                model.to(target_device)
                input_values = input_values.to(target_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(target_device)
                with torch.no_grad():
                    if attention_mask is not None:
                        batch_logits = model(
                            input_values, attention_mask=attention_mask
                        ).logits
                    else:
                        batch_logits = model(input_values).logits
            else:
                raise e

        assert batch_logits is not None

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1)
            feat_extractor = getattr(model, "_get_feat_extract_output_lengths", None)
            if callable(feat_extractor):
                raw_lengths: Any = feat_extractor(input_lengths)
                if hasattr(raw_lengths, "detach"):
                    output_lengths = raw_lengths.detach().cpu().numpy()
                elif isinstance(raw_lengths, np.ndarray):
                    output_lengths = raw_lengths
                else:
                    output_lengths = np.array(raw_lengths)
            else:
                output_lengths = [batch_logits.shape[1]] * batch_logits.shape[0]
        else:
            output_lengths = [batch_logits.shape[1]] * batch_logits.shape[0]

        for b_i, global_idx in enumerate(curr_miss_indices):
            actual_len = int(output_lengths[b_i])
            sliced_logits = batch_logits[b_i, :actual_len, :]
            sliced_lpz = (
                torch.nn.functional.log_softmax(sliced_logits, dim=-1).cpu().numpy()
            )

            item_metadata = dict(metadata)
            sp_len = len(batch_speech[b_i])
            item_metadata["sample_rate"] = TARGET_SAMPLE_RATE
            item_metadata["duration_sec"] = round(sp_len / TARGET_SAMPLE_RATE, 4)

            out_item = ModelOutput(
                lpz=sliced_lpz,
                vocab=vocab,
                frame_duration_sec=FRAME_DURATION_SEC,
                metadata=item_metadata,
            )
            outputs[global_idx] = out_item

            if cache_dir is not None:
                c_dir = Path(cache_dir)
                key = compute_audio_cache_key(
                    audio_inputs[global_idx], model_identifier=str(resolved_model_id)
                )
                try:
                    out_item.save(c_dir / f"{key}.npz")
                except Exception as e:
                    logger.warning("Failed to save ModelOutput cache (%s)", e)

        del batch_logits, inputs, input_values
        if attention_mask is not None:
            del attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return [o for o in outputs if o is not None]
