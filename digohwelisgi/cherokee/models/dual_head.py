# -*- coding: utf-8 -*-
"""
dual_head.py

Split-Layer Dual-Head Wav2Vec2 architecture (`Wav2Vec2ForDualCTC`) and data collator
(`DualDataCollatorCTCWithPadding`) for simultaneous Segmental CV and Moraic Tone CTC.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)
from transformers.modeling_outputs import ModelOutput

from digohwelisgi.cherokee.orthography.tones import MORAIC_TONE_VOCAB
from digohwelisgi.core.audio import AudioAugmenter

TARGET_SAMPLE_RATE = 16000


@dataclass
class Wav2Vec2DualCTCOutput(ModelOutput):
    """
    Output type of `Wav2Vec2ForDualCTC`.

    Attributes:
        loss: Combined scalar multi-task loss (`L_cv + lambda_tone * L_tone`).
        logits: Segmental CV CTC logits of shape `(batch_size, sequence_length, vocab_size)`.
        logits_tone: Moraic tone CTC logits of shape `(batch_size, sequence_length, tone_vocab_size)`.
        hidden_states: Optional tuple of encoder hidden states.
        attentions: Optional tuple of encoder self-attention weights.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    logits_tone: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Wav2Vec2ForDualCTC(Wav2Vec2PreTrainedModel):
    """
    Split-Layer Dual-Head Wav2Vec2 model for multi-task Segmental CV + Moraic Tone CTC.

    Architecture:
    - CNN Feature Extractor + Shared Transformer Trunk (`layers[0 .. N - K - 1]`)
    - Branch A (Segmental CV): `wav2vec2.encoder.layers[N - K .. N - 1]` -> `layer_norm` -> `dropout` -> `lm_head` (19 tokens)
    - Branch B (Moraic Tone): `tone_branch_layers[0 .. K - 1]` -> `tone_layer_norm` -> `tone_dropout` -> `tone_lm_head` (10 tokens)

    By keeping Branch A inside `self.wav2vec2.encoder.layers`, pretrained weight keys
    from standard `Wav2Vec2Model` or `Wav2Vec2ForCTC` checkpoints map 1-to-1, and
    `sync_tone_branch_from_trunk()` initializes Branch B as an exact clone of the
    top `K` pretrained Transformer layers.
    """

    def __init__(
        self,
        config: Any,
        num_branch_layers: Optional[int] = None,
        tone_vocab_size: Optional[int] = None,
        tone_pad_token_id: Optional[int] = None,
        tone_loss_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head."
            )

        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        self.num_branch_layers: int = int(
            num_branch_layers
            if num_branch_layers is not None
            else getattr(config, "num_branch_layers", 2)
        )
        self.tone_vocab_size: int = int(
            tone_vocab_size
            if tone_vocab_size is not None
            else getattr(config, "tone_vocab_size", len(MORAIC_TONE_VOCAB))
        )
        self.tone_pad_token_id: int = int(
            tone_pad_token_id
            if tone_pad_token_id is not None
            else getattr(config, "tone_pad_token_id", MORAIC_TONE_VOCAB["[PAD]"])
        )
        self.tone_loss_weight: float = float(
            tone_loss_weight
            if tone_loss_weight is not None
            else getattr(config, "tone_loss_weight", 1.0)
        )

        # Persist dual-head hyperparameters into config for serialization
        config.num_branch_layers = self.num_branch_layers
        config.tone_vocab_size = self.tone_vocab_size
        config.tone_pad_token_id = self.tone_pad_token_id
        config.tone_loss_weight = self.tone_loss_weight

        num_total_layers = len(self.wav2vec2.encoder.layers)
        if self.num_branch_layers < 0 or self.num_branch_layers > num_total_layers:
            raise ValueError(
                f"num_branch_layers ({self.num_branch_layers}) must be between 0 and {num_total_layers}"
            )
        self.branch_split_idx: int = num_total_layers - self.num_branch_layers

        # Branch B (Moraic Tone): dedicated top K Transformer layers + layer norm + head
        self.tone_branch_layers = nn.ModuleList(
            [
                copy.deepcopy(layer)
                for layer in self.wav2vec2.encoder.layers[self.branch_split_idx :]
            ]
        )
        if hasattr(self.wav2vec2.encoder, "layer_norm"):
            self.tone_layer_norm: Optional[nn.Module] = copy.deepcopy(
                self.wav2vec2.encoder.layer_norm
            )
        else:
            self.tone_layer_norm = None

        self.tone_dropout = nn.Dropout(config.final_dropout)
        self.tone_lm_head = nn.Linear(output_hidden_size, self.tone_vocab_size)

        # Diagnostic tracking of unweighted component losses from the latest forward pass
        self.last_loss_cv: float = 0.0
        self.last_loss_tone: float = 0.0

        self.post_init()

    def sync_tone_branch_from_trunk(self) -> None:
        """
        Clone the top `num_branch_layers` encoder layers and `layer_norm` from `self.wav2vec2.encoder`
        into `self.tone_branch_layers` and `self.tone_layer_norm`.
        Called automatically when initializing from a standard single-head or SSL Wav2Vec2 checkpoint.
        """
        src_layers = self.wav2vec2.encoder.layers[self.branch_split_idx :]
        for dst_layer, src_layer in zip(self.tone_branch_layers, src_layers):
            dst_layer.load_state_dict(src_layer.state_dict())
        if self.tone_layer_norm is not None and hasattr(
            self.wav2vec2.encoder, "layer_norm"
        ):
            self.tone_layer_norm.load_state_dict(
                self.wav2vec2.encoder.layer_norm.state_dict()
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, Any]],
        *model_args: Any,
        **kwargs: Any,
    ) -> "Wav2Vec2ForDualCTC":
        caller_requested_loading_info = bool(kwargs.get("output_loading_info", False))
        kwargs["output_loading_info"] = True

        loaded = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model, loading_info = loaded  # type: ignore[misc]

        missing_keys: List[str] = loading_info.get("missing_keys", [])
        if any(k.startswith("tone_branch_layers.") for k in missing_keys):
            model.sync_tone_branch_from_trunk()

        if caller_requested_loading_info:
            return loaded  # type: ignore[return-value]
        return model

    def freeze_feature_encoder(self) -> None:
        """
        Disable gradient computation for the CNN feature extractor so its parameters
        are not updated during fine-tuning.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_feature_extractor(self) -> None:
        """Alias for `freeze_feature_encoder`."""
        self.freeze_feature_encoder()

    def _prepare_extended_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        sub_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask, add_adapter=False  # type: ignore[arg-type]
        )
        extended_attention_mask = 1.0 - sub_attention_mask[:, None, None, :].to(
            dtype=hidden_states.dtype
        )
        extended_attention_mask = (
            extended_attention_mask * torch.finfo(hidden_states.dtype).min
        )
        return extended_attention_mask

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[Any, ...], Wav2Vec2DualCTCOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Always request hidden_states from self.wav2vec2 so we can tap the Shared Trunk output
        # at `self.branch_split_idx` before Branch A's top K layers run.
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Branch A (Segmental CV): uses full self.wav2vec2 output (layers 0 .. N-1 + layer_norm)
        hidden_states_cv = self.dropout(outputs[0])
        logits = self.lm_head(hidden_states_cv)

        # Branch B (Moraic Tone): taps Shared Trunk at `branch_split_idx` (output of layer N - K - 1)
        if self.num_branch_layers > 0 and outputs.hidden_states is not None:
            tone_hidden = outputs.hidden_states[self.branch_split_idx]
            extended_attention_mask = self._prepare_extended_attention_mask(
                tone_hidden, attention_mask
            )
            for layer in self.tone_branch_layers:
                dropout_probability = torch.rand([])
                skip_the_layer = self.training and (
                    dropout_probability < self.config.layerdrop
                )
                if not skip_the_layer:
                    layer_outputs = layer(
                        tone_hidden,
                        attention_mask=extended_attention_mask,
                        output_attentions=False,
                    )
                    tone_hidden = layer_outputs[0]
            if self.tone_layer_norm is not None:
                tone_hidden = self.tone_layer_norm(tone_hidden)
        else:
            tone_hidden = outputs[0]

        tone_hidden = self.tone_dropout(tone_hidden)
        logits_tone = self.tone_lm_head(tone_hidden)

        loss: Optional[torch.Tensor] = None
        loss_cv: Optional[torch.Tensor] = None
        loss_tone: Optional[torch.Tensor] = None

        if labels is not None or tone_labels is not None:
            if attention_mask is not None:
                raw_lengths: Any = self._get_feat_extract_output_lengths(
                    attention_mask.detach().cpu().sum(-1).to(torch.long)  # type: ignore[arg-type]
                )
                input_lengths = torch.clamp(
                    raw_lengths.to(dtype=torch.long, device=logits.device),
                    min=1,
                    max=logits.shape[1],
                )
            else:
                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=logits.device,
                )

            use_cpu_ctc = logits.device.type == "mps"

            if labels is not None:
                if labels.numel() > 0 and labels.max() >= self.config.vocab_size:
                    raise ValueError(
                        f"Label values must be < vocab_size: {self.config.vocab_size}"
                    )
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                log_probs_cv = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32
                ).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    if use_cpu_ctc:
                        loss_cv = nn.functional.ctc_loss(
                            log_probs_cv.cpu(),
                            flattened_targets.cpu(),
                            input_lengths.cpu(),
                            target_lengths.cpu(),
                            blank=self.config.pad_token_id,
                            reduction=self.config.ctc_loss_reduction,
                            zero_infinity=self.config.ctc_zero_infinity,
                        ).to(logits.device)
                    else:
                        loss_cv = nn.functional.ctc_loss(
                            log_probs_cv,
                            flattened_targets,
                            input_lengths,
                            target_lengths,
                            blank=self.config.pad_token_id,
                            reduction=self.config.ctc_loss_reduction,
                            zero_infinity=self.config.ctc_zero_infinity,
                        )
                self.last_loss_cv = float(loss_cv.detach().cpu().item())

            if tone_labels is not None:
                valid_tone_rows = (tone_labels != -100).any(dim=-1)
                if valid_tone_rows.any():
                    valid_tone_labels = tone_labels[valid_tone_rows]
                    if (
                        valid_tone_labels.numel() > 0
                        and valid_tone_labels.max() >= self.tone_vocab_size
                    ):
                        raise ValueError(
                            f"Tone label values must be < tone_vocab_size: {self.tone_vocab_size}"
                        )
                    tone_mask = valid_tone_labels >= 0
                    tone_target_lengths = tone_mask.sum(-1)
                    flattened_tone_targets = valid_tone_labels.masked_select(tone_mask)
                    valid_input_lengths = input_lengths[valid_tone_rows]

                    log_probs_tone = nn.functional.log_softmax(
                        logits_tone[valid_tone_rows], dim=-1, dtype=torch.float32
                    ).transpose(0, 1)

                    with torch.backends.cudnn.flags(enabled=False):
                        if use_cpu_ctc:
                            loss_tone = nn.functional.ctc_loss(
                                log_probs_tone.cpu(),
                                flattened_tone_targets.cpu(),
                                valid_input_lengths.cpu(),
                                tone_target_lengths.cpu(),
                                blank=self.tone_pad_token_id,
                                reduction=self.config.ctc_loss_reduction,
                                zero_infinity=self.config.ctc_zero_infinity,
                            ).to(logits.device)
                        else:
                            loss_tone = nn.functional.ctc_loss(
                                log_probs_tone,
                                flattened_tone_targets,
                                valid_input_lengths,
                                tone_target_lengths,
                                blank=self.tone_pad_token_id,
                                reduction=self.config.ctc_loss_reduction,
                                zero_infinity=self.config.ctc_zero_infinity,
                            )
                    self.last_loss_tone = float(loss_tone.detach().cpu().item())
                else:
                    # Zero-gradient graph connection when all batch items are unannotated (e.g. Bible-only batch)
                    loss_tone = logits_tone.sum() * 0.0
                    self.last_loss_tone = 0.0

            if loss_cv is not None and loss_tone is not None:
                loss = loss_cv + self.tone_loss_weight * loss_tone
            elif loss_cv is not None:
                loss = loss_cv
            elif loss_tone is not None:
                loss = self.tone_loss_weight * loss_tone

        returned_hidden_states = outputs.hidden_states if output_hidden_states else None

        if not return_dict:
            output: Tuple[Any, ...] = (logits, logits_tone)
            if returned_hidden_states is not None:
                output = output + (returned_hidden_states,)
            if outputs.attentions is not None:
                output = output + (outputs.attentions,)
            return ((loss,) + output) if loss is not None else output

        return Wav2Vec2DualCTCOutput(
            loss=loss,  # type: ignore[arg-type]
            logits=logits,  # type: ignore[arg-type]
            logits_tone=logits_tone,  # type: ignore[arg-type]
            hidden_states=returned_hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DualDataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads `input_values`, `labels` (CV IDs),
    and `tone_labels` (Moraic Tone IDs) with `-100` for masked multi-task CTC training.
    Supports live audio augmentation during training.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    augmenter: Optional[AudioAugmenter] = None
    augment: bool = False

    def __call__(
        self,
        features: List[
            Dict[str, Union[List[int], List[float], np.ndarray, torch.Tensor]]
        ],
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]

        if self.augment and self.augmenter is not None:
            for feature in input_features:
                audio_array = np.asarray(feature["input_values"], dtype=np.float32)
                augmented_audio = self.augmenter(
                    samples=audio_array, sample_rate=TARGET_SAMPLE_RATE
                )
                mean = np.mean(augmented_audio)
                var = np.var(augmented_audio)
                if var > 0:
                    augmented_audio = (augmented_audio - mean) / float(
                        np.sqrt(var + 1e-7)
                    )
                feature["input_values"] = augmented_audio.tolist()

        feat_extractor: Any = getattr(
            self.processor, "feature_extractor", self.processor
        )
        batch = feat_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        if "labels" in features[0]:
            label_tensors = [
                torch.as_tensor(f["labels"], dtype=torch.long) for f in features
            ]
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                label_tensors, batch_first=True, padding_value=-100
            )

        if "tone_labels" in features[0]:
            tone_tensors = [
                torch.as_tensor(
                    f["tone_labels"] if len(f["tone_labels"]) > 0 else [-100],
                    dtype=torch.long,
                )
                for f in features
            ]
            batch["tone_labels"] = torch.nn.utils.rnn.pad_sequence(
                tone_tensors, batch_first=True, padding_value=-100
            )

        return dict(batch)


__all__ = [
    "Wav2Vec2DualCTCOutput",
    "Wav2Vec2ForDualCTC",
    "DualDataCollatorCTCWithPadding",
]
