# -*- coding: utf-8 -*-
"""
model.py

Clean generic ASRModel wrapper holding model, processor, device, and optional caching.
Exposes clean infer() and infer_batch() returning ModelOutput without pass-through delegation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from digohwelisgi.core.models.inference import (
    TARGET_SAMPLE_RATE,
    infer_emissions,
    infer_emissions_batch,
    infer_emissions_sliding_window,
    preprocess_audio,
)
from digohwelisgi.core.models.output import ModelOutput

logger = logging.getLogger(__name__)


class ASRModel:
    """
    Generic CTC ASR model container encapsulating acoustic model, processor, and execution device.
    Exposes clean infer() and infer_batch() returning ModelOutput universal currency.
    """

    model: Any
    processor: Any
    device: str
    model_name: Optional[str]
    cache_dir: Optional[Path]

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Union[str, torch.device] = "cpu",
        model_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.model: Any = model
        self.processor: Any = processor
        self.device = str(device)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def to(self, device: Union[str, torch.device]) -> ASRModel:
        """
        Move the underlying model to the specified device.
        """
        self.device = str(device)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        return self

    def infer(
        self,
        audio_input: Union[
            str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ModelOutput:
        """
        Execute forward inference on single audio input and return ModelOutput universal currency.

        Args:
            audio_input: Audio path, PCM array, tensor, bytes, or AudioSegment.
            sample_rate: Input sampling rate (default: 16000).
            cache_dir: Optional directory for .npz emission caching (overrides instance cache_dir).

        Returns:
            ModelOutput instance containing lpz, vocab, and decoding projections.
        """
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        return infer_emissions(
            model=self.model,
            processor=self.processor,
            audio_input=audio_input,
            sample_rate=sample_rate,
            device=self.device,
            cache_dir=effective_cache_dir,
            model_identifier=self.model_name,
        )

    def infer_batch(
        self,
        audio_inputs: Sequence[
            Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any]
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        batch_size: int = 16,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> List[ModelOutput]:
        """
        Execute batched forward inference on multiple audio inputs returning List[ModelOutput].

        Args:
            audio_inputs: Sequence of audio inputs.
            sample_rate: Input sampling rate.
            batch_size: Forward pass batch size.
            cache_dir: Optional directory for .npz emission caching (overrides instance cache_dir).

        Returns:
            List of ModelOutput instances.
        """
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        return infer_emissions_batch(
            model=self.model,
            processor=self.processor,
            audio_inputs=audio_inputs,
            sample_rate=sample_rate,
            device=self.device,
            batch_size=batch_size,
            cache_dir=effective_cache_dir,
            model_identifier=self.model_name,
        )

    def infer_sliding_window(
        self,
        audio_input: Union[
            str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any
        ],
        chunk_seconds: float = 30.0,
        margin_seconds: float = 1.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ModelOutput:
        """
        Execute overlapping sliding-window inference with margin trimming for long-form audio.

        Args:
            audio_input: Audio path, PCM array, tensor, bytes, or AudioSegment.
            chunk_seconds: Chunk duration in seconds (default: 30.0).
            margin_seconds: Boundary trim margin in seconds (default: 1.0).
            sample_rate: Input sampling rate (default: 16000).
            cache_dir: Optional directory for .npz emission caching (overrides instance cache_dir).

        Returns:
            ModelOutput instance containing stitched lpz, vocab, and decoding projections.
        """
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        return infer_emissions_sliding_window(
            model=self.model,
            processor=self.processor,
            audio_input=audio_input,
            chunk_seconds=chunk_seconds,
            margin_seconds=margin_seconds,
            sample_rate=sample_rate,
            device=self.device,
            cache_dir=effective_cache_dir,
            model_identifier=self.model_name,
        )

    def as_head(self, head_name: str) -> "VirtualHeadModel":
        """
        Return a VirtualHeadModel exposing the specified auxiliary stream as a primary ASRModel.
        """
        return VirtualHeadModel(parent=self, head_name=head_name)

    @staticmethod
    def preprocess_audio(
        audio_input: Union[
            str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> np.ndarray:
        """Preprocess audio to 16kHz mono float32 ndarray."""
        return preprocess_audio(audio_input, sample_rate=sample_rate)


class VirtualHeadModel(ASRModel):
    """
    Lightweight virtual view over an ASRModel exposing an auxiliary stream
    (such as a secondary CTC head) as if it were a primary ASRModel.
    Forward inference calls delegate directly to the parent model, ensuring
    shared trunk computation and disk cache hits.
    """

    parent: ASRModel
    head_name: str

    def __init__(self, parent: ASRModel, head_name: str) -> None:
        self.parent = parent
        self.head_name = head_name
        model_name = (
            f"{parent.model_name}_{head_name}"
            if parent.model_name
            else f"asr_model_{head_name}"
        )
        super().__init__(
            model=parent.model,
            processor=parent.processor,
            device=parent.device,
            model_name=model_name,
            cache_dir=parent.cache_dir,
        )

    def to(self, device: Union[str, torch.device]) -> "VirtualHeadModel":
        self.parent.to(device)
        self.device = str(device)
        return self

    def infer(
        self,
        audio_input: Union[
            str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ModelOutput:
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        full_output = self.parent.infer(
            audio_input=audio_input,
            sample_rate=sample_rate,
            cache_dir=effective_cache_dir,
        )
        stream = full_output.get_stream(self.head_name)
        if stream is None:
            raise ValueError(
                f"Underlying model '{self.parent.model_name}' did not produce stream '{self.head_name}'"
            )
        return stream

    def infer_batch(
        self,
        audio_inputs: Sequence[
            Union[str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any]
        ],
        sample_rate: int = TARGET_SAMPLE_RATE,
        batch_size: int = 16,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> List[ModelOutput]:
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        full_outputs = self.parent.infer_batch(
            audio_inputs=audio_inputs,
            sample_rate=sample_rate,
            batch_size=batch_size,
            cache_dir=effective_cache_dir,
        )
        results: List[ModelOutput] = []
        for out in full_outputs:
            stream = out.get_stream(self.head_name)
            if stream is None:
                raise ValueError(
                    f"Underlying model '{self.parent.model_name}' did not produce stream '{self.head_name}'"
                )
            results.append(stream)
        return results

    def infer_sliding_window(
        self,
        audio_input: Union[
            str, Path, bytes, Sequence[float], np.ndarray, torch.Tensor, Any
        ],
        chunk_seconds: float = 30.0,
        margin_seconds: float = 1.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ModelOutput:
        effective_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        full_output = self.parent.infer_sliding_window(
            audio_input=audio_input,
            chunk_seconds=chunk_seconds,
            margin_seconds=margin_seconds,
            sample_rate=sample_rate,
            cache_dir=effective_cache_dir,
        )
        stream = full_output.get_stream(self.head_name)
        if stream is None:
            raise ValueError(
                f"Underlying model '{self.parent.model_name}' did not produce stream '{self.head_name}'"
            )
        return stream
