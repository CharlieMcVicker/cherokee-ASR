# -*- coding: utf-8 -*-
"""
output.py

Universal rich data container for CTC ASR model emissions and decoding projections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True, init=False)
class ModelOutput:
    """
    Universal rich data container holding raw acoustic model emissions (lpz),
    associated vocabulary/token mappings, frame timing, and pure projection methods.
    Supports arbitrary named auxiliary streams (such as secondary CTC heads) where
    each auxiliary stream is itself an instance of ModelOutput.

    Attributes:
        lpz: Log-probabilities matrix of shape (T, V) or (B, T, V).
             For single utterance inference, typically (T, V).
        vocab: Dictionary mapping token strings to integer IDs or integer IDs to strings.
        frame_duration_sec: Duration of each acoustic frame in seconds (default: 0.02s / 20ms).
        metadata: Optional dictionary holding arbitrary contextual data (e.g. sample_rate, duration).
        auxiliary_streams: Mapping of stream name to auxiliary ModelOutput instances.
    """

    lpz: np.ndarray
    vocab: Mapping[str, int] | Mapping[int, str]
    frame_duration_sec: float
    metadata: Dict[str, Any]
    auxiliary_streams: Dict[str, ModelOutput]

    def __init__(
        self,
        lpz: np.ndarray,
        vocab: Mapping[str, int] | Mapping[int, str],
        frame_duration_sec: float = 0.02,
        metadata: Optional[Dict[str, Any]] = None,
        auxiliary_streams: Optional[Dict[str, ModelOutput]] = None,
        lpz_tone: Optional[np.ndarray] = None,
        vocab_tone: Optional[Mapping[str, int] | Mapping[int, str]] = None,
    ) -> None:
        if not isinstance(lpz, np.ndarray):
            lpz = np.asarray(lpz)
        object.__setattr__(self, "lpz", lpz)
        object.__setattr__(self, "vocab", vocab)
        object.__setattr__(self, "frame_duration_sec", frame_duration_sec)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})

        streams: Dict[str, ModelOutput] = (
            dict(auxiliary_streams) if auxiliary_streams else {}
        )
        if lpz_tone is not None and "tone" not in streams:
            v_tone = vocab_tone if vocab_tone is not None else {}
            streams["tone"] = ModelOutput(
                lpz=lpz_tone,
                vocab=v_tone,
                frame_duration_sec=frame_duration_sec,
            )
        object.__setattr__(self, "auxiliary_streams", streams)

    @property
    def tone(self) -> Optional[ModelOutput]:
        """Convenience property for auxiliary tone stream if present."""
        return self.auxiliary_streams.get("tone")

    def get_stream(self, name: str) -> Optional[ModelOutput]:
        """Retrieve an auxiliary emission stream by name."""
        return self.auxiliary_streams.get(name)

    @property
    def lpz_tone(self) -> Optional[np.ndarray]:
        """Backwards-compatible accessor for tone log-probabilities."""
        t = self.tone
        return t.lpz if t is not None else None

    @property
    def vocab_tone(self) -> Optional[Mapping[str, int] | Mapping[int, str]]:
        """Backwards-compatible accessor for tone vocabulary."""
        t = self.tone
        return t.vocab if t is not None else None

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Resolve mapping from integer token ID to token string."""
        if not self.vocab:
            return {}
        try:
            sample_key = next(iter(self.vocab.keys()))
        except (StopIteration, TypeError):
            return {}
        if isinstance(sample_key, (int, np.integer)):
            return {int(k): str(v) for k, v in self.vocab.items()}  # type: ignore
        else:
            return {int(v): str(k) for k, v in self.vocab.items()}  # type: ignore

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Resolve mapping from token string to integer token ID."""
        if not self.vocab:
            return {}
        try:
            sample_key = next(iter(self.vocab.keys()))
        except (StopIteration, TypeError):
            return {}
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

    @property
    def tone_id_to_token(self) -> Dict[int, str]:
        """Resolve mapping from integer tone token ID to moraic tone token string."""
        if not self.vocab_tone:
            return {}
        sample_key = next(iter(self.vocab_tone.keys()))
        if isinstance(sample_key, (int, np.integer)):
            return {int(k): str(v) for k, v in self.vocab_tone.items()}  # type: ignore
        else:
            return {int(v): str(k) for k, v in self.vocab_tone.items()}  # type: ignore

    @property
    def tone_token_to_id(self) -> Dict[str, int]:
        """Resolve mapping from moraic tone token string to integer tone token ID."""
        if not self.vocab_tone:
            return {}
        sample_key = next(iter(self.vocab_tone.keys()))
        if isinstance(sample_key, str):
            return {str(k): int(v) for k, v in self.vocab_tone.items()}  # type: ignore
        else:
            return {str(v): int(k) for k, v in self.vocab_tone.items()}  # type: ignore

    @property
    def tone_pad_token_id(self) -> int:
        """Resolve pad token ID for moraic tone vocabulary."""
        if "tone_pad_token_id" in self.metadata:
            return int(self.metadata["tone_pad_token_id"])
        t2i = self.tone_token_to_id
        for pad_sym in ("[PAD]", "<pad>"):
            if pad_sym in t2i:
                return t2i[pad_sym]
        return 0

    def decode_tone_tokens(
        self, collapse_repeats: bool = True, remove_pad: bool = True
    ) -> str:
        """
        Pure projection: Greedily decodes `lpz_tone` into a space-separated moraic tone token string
        (e.g. 'C 2_1 | C 3_1 4_2'). Returns empty string if `lpz_tone` or `vocab_tone` is absent.
        """
        if self.lpz_tone is None or not self.vocab_tone:
            return ""
        emissions = self.lpz_tone
        if emissions.ndim == 3:
            if emissions.shape[0] == 1:
                emissions = emissions[0]
            else:
                raise ValueError(
                    "decode_tone_tokens requires 2D (T, V_tone) emissions."
                )
        if emissions.shape[0] == 0:
            return ""

        pred_ids = np.argmax(emissions, axis=-1)
        id2tok = self.tone_id_to_token
        pad_id = self.tone_pad_token_id
        unk_id = self.tone_token_to_id.get("[UNK]", -1)

        out_tokens: List[str] = []
        prev_id = -1
        for idx in pred_ids:
            cur_id = int(idx)
            if collapse_repeats and cur_id == prev_id:
                continue
            prev_id = cur_id
            if (remove_pad and cur_id == pad_id) or cur_id == unk_id:
                continue
            tok_str = id2tok.get(cur_id, "")
            if (
                tok_str
                and tok_str != "[UNK]"
                and (not remove_pad or tok_str != "[PAD]")
            ):
                out_tokens.append(tok_str)
        return " ".join(out_tokens)

    def decode_reconciled(
        self,
        mode: str = "dp",
        reconciler: Optional[Callable[[str, str, str], str]] = None,
    ) -> str:
        """
        Pure projection: Decodes both Segmental CV and Moraic Tone streams and reconciles them
        into a unified numeric-tone string (e.g. 'na3 a2ni22tshu34tsa2').
        Falls back to `.decode_greedy()` if `lpz_tone` is not present.
        """
        cv_text = self.decode_greedy()
        if self.lpz_tone is None or not self.vocab_tone:
            return cv_text
        tone_str = self.decode_tone_tokens(collapse_repeats=True, remove_pad=True)
        if not tone_str:
            return cv_text
        if reconciler is None:
            from digohwelisgi.cherokee.orthography.tones import reconcile_dual_stream

            reconciler = reconcile_dual_stream
        return reconciler(cv_text, tone_str, mode)

    def save(self, path: Union[str, Path]) -> None:
        """
        Persists ModelOutput to a compressed .npz archive, including all auxiliary streams.

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

        save_kwargs: Dict[str, Any] = {
            "lpz": self.lpz.astype(np.float32),
            "vocab_json": np.array(vocab_json),
            "frame_duration_sec": np.array(self.frame_duration_sec, dtype=np.float64),
            "meta_json": np.array(meta_json),
        }
        aux_names = list(self.auxiliary_streams.keys())
        save_kwargs["aux_streams_json"] = np.array(json.dumps(aux_names))
        for name, stream in self.auxiliary_streams.items():
            save_kwargs[f"aux_{name}_lpz"] = stream.lpz.astype(np.float32)
            stream_vocab_json = json.dumps(
                {str(k): int(v) for k, v in stream.token_to_id.items()},
                ensure_ascii=False,
            )
            save_kwargs[f"aux_{name}_vocab_json"] = np.array(stream_vocab_json)
            save_kwargs[f"aux_{name}_meta_json"] = np.array(
                json.dumps(stream.metadata, ensure_ascii=False)
            )

        # Backwards compatibility for existing tools expecting lpz_tone and vocab_tone_json:
        if self.tone is not None:
            save_kwargs["lpz_tone"] = self.tone.lpz.astype(np.float32)
            save_kwargs["vocab_tone_json"] = save_kwargs["aux_tone_vocab_json"]

        np.savez_compressed(dest, **save_kwargs)

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

            auxiliary_streams: Dict[str, ModelOutput] = {}
            if "aux_streams_json" in data:
                aux_names = json.loads(str(data["aux_streams_json"]))
                for name in aux_names:
                    aux_lpz = data[f"aux_{name}_lpz"]
                    aux_vocab = json.loads(str(data[f"aux_{name}_vocab_json"]))
                    aux_meta = (
                        json.loads(str(data[f"aux_{name}_meta_json"]))
                        if f"aux_{name}_meta_json" in data
                        else {}
                    )
                    auxiliary_streams[name] = ModelOutput(
                        lpz=aux_lpz,
                        vocab=aux_vocab,
                        frame_duration_sec=frame_duration,
                        metadata=aux_meta,
                    )
            elif "lpz_tone" in data and "vocab_tone_json" in data:
                # Backwards compatibility for legacy single-archive files
                aux_lpz = data["lpz_tone"]
                aux_vocab = json.loads(str(data["vocab_tone_json"]))
                auxiliary_streams["tone"] = ModelOutput(
                    lpz=aux_lpz,
                    vocab=aux_vocab,
                    frame_duration_sec=frame_duration,
                )

        return cls(
            lpz=lpz,
            vocab=vocab,
            frame_duration_sec=frame_duration,
            metadata=metadata,
            auxiliary_streams=auxiliary_streams,
        )
