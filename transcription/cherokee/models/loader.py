# -*- coding: utf-8 -*-
"""
loader.py

Cherokee ASR model factory and encapsulation loading default Cherokee repositories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from transcription.core.models.model import ASRModel
from transcription.utils.model_utils import get_best_model_config, get_model

logger = logging.getLogger(__name__)

DEFAULT_CHEROKEE_FALLBACK_REPO = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000
FRAME_DURATION_SEC = 0.02


@dataclass
class WordConfidence:
    word: str
    confidence: float
    start_time: float
    end_time: float
    chars: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ASRResult:
    text: str
    transcription: str
    confidence: float
    words: List[WordConfidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "transcription": self.transcription,
            "confidence": self.confidence,
            "words": [
                w.to_dict() if isinstance(w, WordConfidence) else w for w in self.words
            ],
        }


class CherokeeASRModel(ASRModel):
    """
    Cherokee ASR model container wrapping generic ASRModel.
    Provides Cherokee-specific pretrained loading, default HuggingFace repository resolution,
    and fallback mechanics.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Union[str, torch.device] = "cpu",
        model_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            model=model,
            processor=processor,
            device=device,
            model_name=model_name,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        revision: Optional[str] = None,
        processor_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        token: Optional[str] = None,
        eval_mode: bool = True,
        use_cache: bool = True,
    ) -> CherokeeASRModel:
        """
        Instantiate CherokeeASRModel from a Hugging Face repository or local directory path.
        """
        model, processor, resolved_device = get_model(
            path_or_repo=path_or_repo,
            revision=revision,
            processor_path=processor_path,
            device=device,
            token=token,
            eval_mode=eval_mode,
            use_cache=use_cache,
        )
        resolved_name = f"{path_or_repo}_{revision}" if revision else str(path_or_repo)
        return cls(
            model=model,
            processor=processor,
            device=resolved_device,
            model_name=resolved_name,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        token: Optional[str] = None,
        eval_mode: bool = True,
        use_cache: bool = True,
    ) -> CherokeeASRModel:
        """
        Instantiate CherokeeASRModel from a configuration dictionary containing 'repo' and optional 'revision'.
        """
        repo = config.get("repo")
        if not repo:
            raise ValueError("Config must contain 'repo' key.")
        revision = config.get("revision")
        processor_path = config.get("processor_path")
        return cls.from_pretrained(
            path_or_repo=repo,
            revision=revision,
            processor_path=processor_path,
            device=device,
            token=token,
            eval_mode=eval_mode,
            use_cache=use_cache,
        )

    @classmethod
    def get_best_model(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        token: Optional[str] = None,
        eval_mode: bool = True,
        use_cache: bool = True,
    ) -> CherokeeASRModel:
        """
        Instantiate CherokeeASRModel using the best Cherokee model configuration discovered via best_model.json.
        """
        config = get_best_model_config()
        return cls.from_config(
            config=config,
            device=device,
            token=token,
            eval_mode=eval_mode,
            use_cache=use_cache,
        )

    @classmethod
    def from_pretrained_or_best(
        cls,
        path_or_repo: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        fallback_repo: str = DEFAULT_CHEROKEE_FALLBACK_REPO,
        eval_mode: bool = True,
        use_cache: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        processor_path: Optional[str] = None,
    ) -> CherokeeASRModel:
        """
        Unified factory method that instantiates CherokeeASRModel with automatic resolution and fallback.
        """
        if path_or_repo is not None:
            try:
                return cls.from_pretrained(
                    path_or_repo=path_or_repo,
                    revision=revision,
                    processor_path=processor_path,
                    device=device,
                    token=token,
                    eval_mode=eval_mode,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load model from '%s' (%s). Falling back to '%s'.",
                    path_or_repo,
                    e,
                    fallback_repo,
                )
                return cls.from_pretrained(
                    path_or_repo=fallback_repo,
                    device=device,
                    token=token,
                    eval_mode=eval_mode,
                    use_cache=use_cache,
                )
        else:
            try:
                return cls.get_best_model(
                    device=device,
                    token=token,
                    eval_mode=eval_mode,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load best model (%s). Falling back to '%s'.",
                    e,
                    fallback_repo,
                )
                return cls.from_pretrained(
                    path_or_repo=fallback_repo,
                    device=device,
                    token=token,
                    eval_mode=eval_mode,
                    use_cache=use_cache,
                )

    # -------------------------------------------------------------------------
    # Procedural Inference Layers
    # -------------------------------------------------------------------------

    def get_logits(
        self,
        pcm_audio: Union[str, bytes, List[float], np.ndarray, torch.Tensor],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> torch.Tensor:
        """
        Layer 1: Accepts PCM audio input and produces raw logits tensor [sequence_length, vocab_size].
        """
        speech = self.preprocess_audio(pcm_audio, sample_rate=sample_rate)
        input_values = self.processor(
            speech, sampling_rate=TARGET_SAMPLE_RATE
        ).input_values[0]
        input_tensor = torch.tensor(np.array([input_values]), dtype=torch.float32).to(
            self.device
        )

        try:
            with torch.no_grad():
                logits = self.model(input_tensor).logits
        except NotImplementedError as e:
            if self.device == "mps":
                self.device = "cpu"
                self.model.to(self.device)
                input_tensor = input_tensor.to(self.device)
                with torch.no_grad():
                    logits = self.model(input_tensor).logits
            else:
                raise e

        return logits.squeeze(0)

    def get_logits_sliding_window(
        self,
        pcm_audio: Union[str, bytes, List[float], np.ndarray, torch.Tensor],
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_seconds: float = 30.0,
        margin_seconds: float = 1.0,
    ) -> torch.Tensor:
        """
        Extracts raw logits tensor [sequence_length, vocab_size] using overlapping
        sliding-window inference with margin trimming to eliminate boundary artifacts.
        Delegates to Tier 1 core infer_sliding_window.
        """
        output = self.infer_sliding_window(
            audio_input=pcm_audio,
            chunk_seconds=chunk_seconds,
            margin_seconds=margin_seconds,
            sample_rate=sample_rate,
        )
        return torch.tensor(output.lpz, dtype=torch.float32, device=self.device)

    def get_logits_batch(
        self,
        audio_inputs: Sequence[
            Union[str, bytes, List[float], np.ndarray, torch.Tensor]
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        batch_size: int = 16,
    ) -> List[torch.Tensor]:
        """
        Batched procedural pipeline for extracting raw logits tensors for multiple audio inputs.
        Applies dynamic padding and returns unpadded sliced logits tensors [seq_len_i, vocab_size].
        """
        results: List[torch.Tensor] = []
        if not audio_inputs:
            return results

        for i in range(0, len(audio_inputs), batch_size):
            batch = audio_inputs[i : i + batch_size]
            batch_speech = []
            for item in batch:
                try:
                    speech = self.preprocess_audio(item, sample_rate=sample_rate)
                    batch_speech.append(speech)
                except Exception as e:
                    logger.warning("Error preprocessing batch item: %s", e)
                    batch_speech.append(np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32))

            inputs = self.processor(
                batch_speech,
                sampling_rate=TARGET_SAMPLE_RATE,
                padding=True,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            try:
                with torch.no_grad():
                    if attention_mask is not None:
                        batch_logits = self.model(
                            input_values, attention_mask=attention_mask
                        ).logits
                    else:
                        batch_logits = self.model(input_values).logits
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or (
                    hasattr(torch.cuda, "OutOfMemoryError")
                    and isinstance(e, torch.cuda.OutOfMemoryError)
                )
                if is_oom or "cudnn" in err_str:
                    logger.warning(
                        "OOM or cuDNN error in batch. Falling back to sequential."
                    )
                    del input_values
                    if attention_mask is not None:
                        del attention_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                    for sp in batch_speech:
                        results.append(
                            self.get_logits(sp, sample_rate=TARGET_SAMPLE_RATE)
                        )
                    continue
                elif isinstance(e, NotImplementedError) and self.device == "mps":
                    logger.warning("MPS error in batch. Falling back to CPU.")
                    self.device = "cpu"
                    self.model.to(self.device)
                    input_values = input_values.to(self.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    with torch.no_grad():
                        if attention_mask is not None:
                            batch_logits = self.model(
                                input_values, attention_mask=attention_mask
                            ).logits
                        else:
                            batch_logits = self.model(input_values).logits
                else:
                    raise e

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1)
                feat_extractor = getattr(
                    self.model, "_get_feat_extract_output_lengths", None
                )
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

            for idx in range(batch_logits.shape[0]):
                actual_len = int(output_lengths[idx])
                sliced_logits = batch_logits[idx, :actual_len, :]
                results.append(sliced_logits)

            # Cleanup batch GPU memory
            del batch_logits, inputs, input_values
            if attention_mask is not None:
                del attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return results

    def get_probabilities(
        self,
        logits_or_audio: Union[torch.Tensor, np.ndarray, str, bytes, List[float]],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> torch.Tensor:
        """
        Layer 2: Obtains softmax probabilities [sequence_length, vocab_size] from logits or audio.
        """
        if isinstance(logits_or_audio, torch.Tensor) and logits_or_audio.ndim >= 2:
            logits = logits_or_audio
        elif isinstance(logits_or_audio, np.ndarray) and logits_or_audio.ndim >= 2:
            logits = torch.tensor(logits_or_audio)
        else:
            logits = self.get_logits(logits_or_audio, sample_rate=sample_rate)

        return torch.nn.functional.softmax(logits, dim=-1)

    def get_word_confidences(
        self,
        probs_or_logits: Union[torch.Tensor, np.ndarray],
    ) -> List[Any]:
        """
        Layer 3: Computes word-level and character-level confidence details with timestamps.
        """
        if isinstance(probs_or_logits, torch.Tensor):
            tensor_data = probs_or_logits.detach().cpu()
        else:
            tensor_data = torch.tensor(probs_or_logits)

        if tensor_data.ndim == 3:
            tensor_data = tensor_data.squeeze(0)

        row_sums = torch.sum(tensor_data, dim=-1)
        if torch.any(tensor_data < 0) or not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2
        ):
            probs = torch.nn.functional.softmax(tensor_data, dim=-1).numpy()
        else:
            probs = tensor_data.numpy()

        pred_ids = np.argmax(probs, axis=-1)
        token_probs = probs[np.arange(len(pred_ids)), pred_ids]

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.processor.tokenizer.vocab.get("[PAD]", 0)

        chars = []
        char_probs = []
        char_alts = []
        char_times = []

        top_k = 5
        top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)

        word_delimiter_token_id = getattr(
            self.processor.tokenizer, "word_delimiter_token_id", None
        )

        prev_id = -1
        for i, token_id in enumerate(pred_ids):
            if token_id != pad_id and token_id != prev_id:
                if token_id == word_delimiter_token_id:
                    token_str = " "
                else:
                    token_str = self.processor.decode([token_id])

                if token_str:
                    chars.append(token_str)
                    char_probs.append(float(token_probs[i]))
                    char_times.append(round(i * 0.02, 3))

                    alts = []
                    for k in range(top_k):
                        alt_id = top_k_indices[i, k]
                        alt_prob = float(top_k_probs[i, k])
                        if alt_id != token_id:
                            if alt_id == word_delimiter_token_id:
                                alt_str = " "
                            else:
                                alt_str = self.processor.decode([alt_id])
                            if alt_str:
                                alts.append({"char": alt_str, "confidence": alt_prob})
                    char_alts.append(alts)

            prev_id = token_id

        words_details: List[WordConfidence] = []
        current_word = ""
        current_chars: List[Dict[str, Any]] = []

        for char, prob, alts, time in zip(chars, char_probs, char_alts, char_times):
            if char == " ":
                if current_word:
                    word_conf = float(np.mean([c["confidence"] for c in current_chars]))
                    words_details.append(
                        WordConfidence(
                            word=current_word,
                            confidence=word_conf,
                            start_time=current_chars[0]["start_time"],
                            end_time=round(current_chars[-1]["start_time"] + 0.02, 3),
                            chars=current_chars,
                        )
                    )
                    current_word = ""
                    current_chars = []
            else:
                current_word += char
                current_chars.append(
                    {
                        "char": char,
                        "confidence": prob,
                        "start_time": time,
                        "alternatives": alts,
                    }
                )

        if current_word:
            word_conf = float(np.mean([c["confidence"] for c in current_chars]))
            words_details.append(
                WordConfidence(
                    word=current_word,
                    confidence=word_conf,
                    start_time=current_chars[0]["start_time"],
                    end_time=round(current_chars[-1]["start_time"] + 0.02, 3),
                    chars=current_chars,
                )
            )

        return words_details

    def decode(
        self,
        logits_or_probs: Union[torch.Tensor, np.ndarray],
        compute_word_confidences: bool = True,
    ) -> Any:
        """
        Layer 4: Decodes logits or probabilities tensor into structured ASRResult.
        """
        if isinstance(logits_or_probs, np.ndarray):
            tensor_data = torch.tensor(logits_or_probs)
        else:
            tensor_data = logits_or_probs

        if tensor_data.ndim == 3:
            tensor_data = tensor_data.squeeze(0)

        row_sums = torch.sum(tensor_data, dim=-1)
        if torch.any(tensor_data < 0) or not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2
        ):
            probs = torch.nn.functional.softmax(tensor_data, dim=-1)
        else:
            probs = tensor_data

        pred_ids = torch.argmax(probs, dim=-1)
        if hasattr(self.processor, "batch_decode"):
            decoded_text = self.processor.batch_decode(pred_ids.unsqueeze(0))[0].strip()
        else:
            decoded_text = self.processor.decode(pred_ids).strip()

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.processor.tokenizer.vocab.get("[PAD]", 0)

        seq_probs = probs.detach().cpu().numpy()
        seq_ids = pred_ids.detach().cpu().numpy()
        token_probs = seq_probs[np.arange(len(seq_ids)), seq_ids]
        non_pad_mask = seq_ids != pad_id

        if non_pad_mask.any():
            confidence = float(np.mean(token_probs[non_pad_mask]))
        else:
            confidence = float(np.mean(token_probs)) if len(token_probs) > 0 else 0.0

        words = self.get_word_confidences(probs) if compute_word_confidences else []

        return ASRResult(
            text=decoded_text,
            transcription=decoded_text,
            confidence=confidence,
            words=words,
        )

    def transcribe(
        self,
        audio_input: Union[str, bytes, List[float], np.ndarray, torch.Tensor],
        sample_rate: int = TARGET_SAMPLE_RATE,
        compute_word_confidences: bool = True,
    ) -> Any:
        """
        Full procedural pipeline for single audio input.
        """
        logits = self.get_logits(audio_input, sample_rate=sample_rate)
        return self.decode(logits, compute_word_confidences=compute_word_confidences)

    def transcribe_batch(
        self,
        audio_inputs: Sequence[
            Union[str, bytes, List[float], np.ndarray, torch.Tensor]
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        batch_size: int = 16,
        compute_word_confidences: bool = False,
    ) -> List[Any]:
        """
        Batched procedural pipeline for multiple audio inputs.
        """
        logits_list = self.get_logits_batch(
            audio_inputs=audio_inputs,
            sample_rate=sample_rate,
            batch_size=batch_size,
        )
        return [
            self.decode(logits, compute_word_confidences=compute_word_confidences)
            for logits in logits_list
        ]


def load_cherokee_asr_model(
    path_or_repo: Optional[str] = None,
    revision: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    use_cache: bool = True,
) -> CherokeeASRModel:
    """
    Convenience factory function to load CherokeeASRModel with default Cherokee repositories.
    """
    return CherokeeASRModel.from_pretrained_or_best(
        path_or_repo=path_or_repo,
        revision=revision,
        device=device,
        token=token,
        use_cache=use_cache,
    )


__all__ = [
    "CherokeeASRModel",
    "load_cherokee_asr_model",
    "ASRResult",
    "WordConfidence",
    "DEFAULT_CHEROKEE_FALLBACK_REPO",
]
