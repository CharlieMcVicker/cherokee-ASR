# -*- coding: utf-8 -*-
"""
output.py

Universal rich data container for CTC ASR model emissions and decoding projections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class ModelOutput:
    """
    Universal rich data container holding raw acoustic model emissions (lpz),
    associated vocabulary/token mappings, frame timing, and pure projection methods.

    Attributes:
        lpz: Log-probabilities matrix of shape (T, V) or (B, T, V).
             For single utterance inference, typically (T, V).
        vocab: Dictionary mapping token strings to integer IDs or integer IDs to strings.
        frame_duration_sec: Duration of each acoustic frame in seconds (default: 0.02s / 20ms).
        metadata: Optional dictionary holding arbitrary contextual data (e.g. sample_rate, duration).
    """

    lpz: np.ndarray
    vocab: Mapping[str, int] | Mapping[int, str]
    frame_duration_sec: float = 0.02
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.lpz, np.ndarray):
            object.__setattr__(self, "lpz", np.asarray(self.lpz))

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Resolve mapping from integer token ID to token string."""
        if not self.vocab:
            return {}
        # Check if keys are integers
        sample_key = next(iter(self.vocab.keys()))
        if isinstance(sample_key, (int, np.integer)):
            return {int(k): str(v) for k, v in self.vocab.items()}  # type: ignore
        else:
            return {int(v): str(k) for k, v in self.vocab.items()}  # type: ignore

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Resolve mapping from token string to integer token ID."""
        if not self.vocab:
            return {}
        sample_key = next(iter(self.vocab.keys()))
        if isinstance(sample_key, str):
            return {str(k): int(v) for k, v in self.vocab.items()}  # type: ignore
        else:
            return {str(v): int(k) for k, v in self.vocab.items()}  # type: ignore

    @property
    def pad_token_id(self) -> int:
        """Resolve pad token ID (checks metadata, vocab '[PAD]', '<pad>', or defaults to 0)."""
        if "pad_token_id" in self.metadata:
            return int(self.metadata["pad_token_id"])
        t2i = self.token_to_id
        for pad_sym in ("[PAD]", "<pad>"):
            if pad_sym in t2i:
                return t2i[pad_sym]
        return 0

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """Resolve word delimiter token ID (e.g. '|', ' ')."""
        if "word_delimiter_token_id" in self.metadata:
            val = self.metadata["word_delimiter_token_id"]
            return int(val) if val is not None else None
        t2i = self.token_to_id
        for delim_sym in ("|", " "):
            if delim_sym in t2i:
                return t2i[delim_sym]
        return None

    def decode_tokens(
        self, collapse_repeats: bool = True, remove_pad: bool = True
    ) -> List[str]:
        """
        Pure projection: Decodes argmax emission indices into a sequence of token strings.

        Args:
            collapse_repeats: Whether to collapse consecutive identical token IDs (CTC rule).
            remove_pad: Whether to filter out blank/pad tokens.

        Returns:
            List of decoded token strings for each non-blank/collapsed frame.
        """
        emissions = self.lpz
        if emissions.ndim == 3:
            # If batched with batch_size=1, squeeze
            if emissions.shape[0] == 1:
                emissions = emissions[0]
            else:
                raise ValueError(
                    "decode_tokens requires 2D (T, V) emissions. For batch, index into batch first."
                )

        if emissions.shape[0] == 0:
            return []

        pred_ids = np.argmax(emissions, axis=-1)
        id2tok = self.id_to_token
        pad_id = self.pad_token_id
        delim_id = self.word_delimiter_token_id

        tokens: List[str] = []
        prev_id = -1
        for idx in pred_ids:
            cur_id = int(idx)
            if collapse_repeats and cur_id == prev_id:
                continue
            prev_id = cur_id
            if remove_pad and cur_id == pad_id:
                continue
            if delim_id is not None and cur_id == delim_id:
                tokens.append(" ")
            else:
                tok_str = id2tok.get(cur_id, "")
                if tok_str:
                    tokens.append(tok_str)

        return tokens

    def decode_greedy(self) -> str:
        """
        Pure projection: Greedily decodes emissions into a unified text string.
        Collapses CTC consecutive repeats, strips pads, replaces delimiters with space,
        and cleanly normalizes whitespace.

        Returns:
            Greedy transcribed string.
        """
        tokens = self.decode_tokens(collapse_repeats=True, remove_pad=True)
        raw_text = "".join(tokens)
        return " ".join(raw_text.split())

    def save(self, path: Union[str, Path]) -> None:
        """
        Persists ModelOutput to a compressed .npz archive.

        Args:
            path: Destination file path (appends .npz if not present).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        vocab_json = json.dumps(
            {str(k): int(v) for k, v in self.token_to_id.items()},
            ensure_ascii=False,
        )
        meta_json = json.dumps(self.metadata, ensure_ascii=False)

        np.savez_compressed(
            dest,
            lpz=self.lpz.astype(np.float32),
            vocab_json=np.array(vocab_json),
            frame_duration_sec=np.array(self.frame_duration_sec, dtype=np.float64),
            meta_json=np.array(meta_json),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> ModelOutput:
        """
        Loads a ModelOutput instance from a compressed .npz archive.

        Args:
            path: Path to the .npz archive.

        Returns:
            Deserialized ModelOutput instance.
        """
        src = Path(path)
        if not src.exists() and not str(src).endswith(".npz"):
            src = Path(f"{src}.npz")

        with np.load(src, allow_pickle=False) as data:
            lpz = data["lpz"]
            vocab_str = str(data["vocab_json"])
            vocab = json.loads(vocab_str)
            frame_duration = (
                float(data["frame_duration_sec"])
                if "frame_duration_sec" in data
                else 0.02
            )
            metadata = json.loads(str(data["meta_json"])) if "meta_json" in data else {}

        return cls(
            lpz=lpz,
            vocab=vocab,
            frame_duration_sec=frame_duration,
            metadata=metadata,
        )
