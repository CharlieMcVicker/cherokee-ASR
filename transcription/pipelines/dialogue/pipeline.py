# -*- coding: utf-8 -*-
"""
transcription.pipelines.dialogue.pipeline module.

End-to-end Code-Switched Dialogue Alignment Pipeline.
Orchestrates:
1. Ingestion of multi-speaker dialogue text transcripts with speaker prefixes (e.g., 'Guy Soldier: ...', 'Charley: ...').
2. Token discrimination via CodeSwitchedPreparer from transcription.cherokee.codeswitching,
   isolating English loanwords/names and compound clitics (e.g., 'JayᎢ') from Cherokee syncope/intrusion masks.
3. Continuous acoustic inference via CherokeeASRModel / ModelOutput universal currency.
4. Optional Silero VAD soft-masking via transcription.core.audio.masking.
5. Syncope- and intrusion-aware CTC segmentation via CTCSegmentationAligner (transcription.core.alignment.ctc).
6. Phonetic syllabary reconciliation via transcription.cherokee.enrichment.
7. Assembling and exporting canonical 7-tier Praat TextGrids:
   - Turn (or Chunks)
   - Speaker
   - Syllabary (Syllabary Words)
   - English (English Words)
   - Reconciled (Reconciled Words)
   - CTC Word (Words / Padded Words)
   - Phoneme
   and structured JSON manifests via transcription.core.exporters.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydub import AudioSegment

from transcription.core.alignment.ctc import DEFAULT_CACHE_DIR


from transcription.cherokee.codeswitching import (
    CodeSwitchedLineResult,
    CodeSwitchedPreparer,
    CodeSwitchedToken,
    TokenType,
    create_groundtruth_for_code_switched_syllabary,
    extract_speaker_prefix,
    get_default_projector,
)
from transcription.cherokee.enrichment import (
    align_character_syllable,
    reconcile_alignment_words,
    reconcile_phonetics,
    reconcile_word_intervals,
)
from transcription.cherokee.models import CherokeeASRModel
from transcription.cherokee.phonotactics import (
    PhonemeCategory,
    PhonotacticToken,
    prepare_cherokee_text,
    tokenize_phonemes,
)
from transcription.core.alignment.ctc import (
    CTCSegmentationAligner as CoreCTCSegmentationAligner,
)
from transcription.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.core.audio.masking import (
    SileroVADDetector,
    apply_vad_soft_masking,
    extract_vad_intervals,
    mask_non_speech_logits,
)
from transcription.core.audio.segment import AudioChunk
from transcription.core.exporters.manifest import (
    export_debug_json,
    export_manifest as export_manifest_file,
)
from transcription.core.exporters.textgrid import (
    IntervalTier,
    TextGridBuilder,
    build_contiguous_intervals,
    build_padded_word_intervals,
    export_textgrid as export_textgrid_file,
)
from transcription.core.models.output import ModelOutput

logger = logging.getLogger(__name__)


def build_syllabary_word_tier(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> List[WordInterval]:
    """
    Builds a list of WordIntervals containing Cherokee syllabary tokens
    mapped to aligned word timestamps.
    """
    syllabary_words: List[WordInterval] = []
    for chunk in alignment.aligned_chunks:
        syll_chunk_str = syllabary_lookup.get(chunk.chunk_id, "")
        s_words = syll_chunk_str.split()
        syll_idx = 0
        for w in chunk.words:
            k = max(1, len(w.word.split())) if w.word else 1
            if s_words and syll_idx >= len(s_words):
                raise ValueError(
                    f"Syllabary word index {syll_idx} exceeds token bounds ({len(s_words)}) for chunk '{chunk.chunk_id}'."
                )
            if s_words and syll_idx < len(s_words):
                syll_w = " ".join(s_words[syll_idx : syll_idx + k])
                syll_idx += k
            else:
                syll_w = w.word
            syllabary_words.append(
                WordInterval(
                    word=syll_w,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=w.emitted_word,
                )
            )
    return syllabary_words


def build_english_word_tier(
    alignment: AlignmentOutput,
    source_lookup: Dict[str, Dict[str, Any]],
) -> List[WordInterval]:
    """
    Builds a list of WordIntervals containing English words/stems from code-switched tokens
    mapped to aligned word timestamps. Non-English intervals contain empty strings.
    """
    english_words: List[WordInterval] = []
    for chunk in alignment.aligned_chunks:
        meta = source_lookup.get(chunk.chunk_id, {})
        cs_meta = meta.get("code_switched")
        tokens = cs_meta.get("tokens", []) if isinstance(cs_meta, dict) else []
        tok_idx = 0
        for w in chunk.words:
            k = max(1, len(w.word.split())) if w.word else 1
            if tokens and tok_idx >= len(tokens):
                raise ValueError(
                    f"Code-switched token index {tok_idx} exceeds token bounds ({len(tokens)}) for chunk '{chunk.chunk_id}'."
                )
            eng_parts: List[str] = []
            if tokens and tok_idx < len(tokens):
                for t in tokens[tok_idx : tok_idx + k]:
                    stem = t.get("english_stem")
                    if stem:
                        eng_parts.append(stem)
                tok_idx += k
            eng_w = " ".join(eng_parts) if eng_parts else ""
            english_words.append(
                WordInterval(
                    word=eng_w,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=w.emitted_word,
                )
            )
    return english_words


def build_speaker_intervals(
    alignment: AlignmentOutput,
    source_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Builds speaker intervals from chunk boundaries and extracted speaker metadata.
    """
    intervals: List[Dict[str, Any]] = []
    for chunk in alignment.aligned_chunks:
        meta = source_lookup.get(chunk.chunk_id, {})
        speaker = meta.get("speaker") or ""
        intervals.append(
            {
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "text": speaker,
            }
        )
    return intervals


def build_turn_intervals(
    alignment: AlignmentOutput,
    source_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Builds dialogue turn intervals from chunk boundaries and source text/metadata.
    """
    intervals: List[Dict[str, Any]] = []
    for chunk in alignment.aligned_chunks:
        meta = source_lookup.get(chunk.chunk_id, {})
        text_val = meta.get("text", meta.get("syllabary", ""))
        speaker = meta.get("speaker")
        label = (
            f"{speaker}: {text_val}"
            if speaker and not text_val.startswith(f"{speaker}:")
            else text_val
        )
        if not label:
            label = chunk.chunk_id
        intervals.append(
            {
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "text": label,
            }
        )
    return intervals


def build_phoneme_tier(
    reconciled_or_ctc_words: Sequence[WordInterval],
) -> List[WordInterval]:
    """
    Deconstructs aligned words into sub-word phonetic units / phonemes
    using phonotactic tokenization, linearly apportioning word duration.
    """
    phoneme_intervals: List[WordInterval] = []
    for w in reconciled_or_ctc_words:
        word_text = (w.emitted_word or w.word or "").strip()
        if not word_text:
            continue
        duration = max(0.0, w.end_sec - w.start_sec)
        tokens = tokenize_phonemes(word_text)
        non_space_tokens = [
            t
            for t in tokens
            if t.symbol.strip() and t.category != PhonemeCategory.OTHER
        ]
        if not non_space_tokens:
            # Fallback to single interval for non-analyzed token
            phoneme_intervals.append(
                WordInterval(
                    word=word_text,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=word_text,
                )
            )
            continue

        num_phones = len(non_space_tokens)
        dt = duration / float(num_phones) if num_phones > 0 else 0.0
        cur_t = w.start_sec
        for idx, pt in enumerate(non_space_tokens):
            p_end = cur_t + dt if idx < num_phones - 1 else w.end_sec
            phoneme_intervals.append(
                WordInterval(
                    word=pt.symbol,
                    start_sec=round(cur_t, 3),
                    end_sec=round(p_end, 3),
                    confidence=w.confidence,
                    flagged=w.flagged,
                    emitted_word=pt.symbol,
                )
            )
            cur_t = p_end

    return phoneme_intervals


def export_7tier_textgrid(
    alignment: AlignmentOutput,
    output_dir: Union[str, Path],
    filename: str = "dialogue_alignment.TextGrid",
    source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    syllabary_words: Optional[Sequence[WordInterval]] = None,
    english_words: Optional[Sequence[WordInterval]] = None,
    reconciled_words: Optional[Sequence[WordInterval]] = None,
    phoneme_words: Optional[Sequence[WordInterval]] = None,
    pad_sec: float = 0.10,
    include_legacy_chunks_tier: bool = True,
) -> str:
    """
    Assembles and exports canonical 7-tier Praat TextGrids:
    1. Turn (dialogue turn / chunk label)
    2. Speaker (speaker identity prefix)
    3. Syllabary (Cherokee Syllabary words)
    4. English (English loanwords/names)
    5. Reconciled (reconciled phonetic words)
    6. CTC Word (aligned word bounds)
    7. Phoneme (sub-word phonetic unit bounds)

    Also preserves 'Chunks', 'Words', 'Padded Words', 'Syllabary Words', 'English Words',
    'Reconciled Words' tiers for backwards compatibility with existing Praat visualizers and test fixtures.

    Returns:
        Path to generated TextGrid file.
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = out_dir_path / filename

    meta_lookup = source_metadata or {}

    total_end = 0.0
    if alignment.aligned_chunks:
        total_end = max(c.end_sec for c in alignment.aligned_chunks)
    if alignment.raw_tokens:
        total_end = max(total_end, max(t.end_sec for t in alignment.raw_tokens))
    for seq in (syllabary_words, english_words, reconciled_words, phoneme_words):
        if seq:
            total_end = max(total_end, max(w.end_sec for w in seq))
    if total_end <= 0.0:
        total_end = 1.0

    builder = TextGridBuilder(xmin=0.0, xmax=total_end)

    # 1. Turn tier
    turn_raw = build_turn_intervals(alignment, meta_lookup)
    turn_intervals = build_contiguous_intervals(turn_raw, total_end)
    builder.create_tier("Turn", turn_intervals)
    if include_legacy_chunks_tier:
        builder.create_tier("Chunks", turn_intervals)

    # 2. Speaker tier
    speaker_raw = build_speaker_intervals(alignment, meta_lookup)
    speaker_intervals = build_contiguous_intervals(speaker_raw, total_end)
    builder.create_tier("Speaker", speaker_intervals)

    # Word-level baseline intervals
    word_raw: List[Dict[str, Any]] = []
    for c in alignment.aligned_chunks:
        for w in c.words:
            word_raw.append(
                {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            )
    ctc_word_intervals = build_contiguous_intervals(word_raw, total_end)
    padded_word_intervals = build_padded_word_intervals(
        word_raw, total_end, pad_sec=pad_sec
    )

    # 3. Syllabary tier
    syll_tier_items = (
        [
            {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            for w in syllabary_words
            if w.word and w.word.strip()
        ]
        if syllabary_words
        else []
    )
    syll_intervals = build_contiguous_intervals(syll_tier_items, total_end)
    builder.create_tier("Syllabary", syll_intervals)
    builder.create_tier("Syllabary Words", syll_intervals)

    # 4. English tier
    eng_tier_items = (
        [
            {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            for w in english_words
            if w.word and w.word.strip()
        ]
        if english_words
        else []
    )
    eng_intervals = build_contiguous_intervals(eng_tier_items, total_end)
    builder.create_tier("English", eng_intervals)
    builder.create_tier("English Words", eng_intervals)

    # 5. Reconciled tier
    rec_tier_items = (
        [
            {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            for w in reconciled_words
            if w.word and w.word.strip()
        ]
        if reconciled_words
        else []
    )
    rec_intervals = build_contiguous_intervals(rec_tier_items, total_end)
    builder.create_tier("Reconciled", rec_intervals)
    builder.create_tier("Reconciled Words", rec_intervals)

    # 6. CTC Word tier
    builder.create_tier("CTC Word", ctc_word_intervals)
    builder.create_tier("Words", ctc_word_intervals)
    builder.create_tier("Padded Words", padded_word_intervals)

    # 7. Phoneme tier
    resolved_phonemes = phoneme_words
    if resolved_phonemes is None and reconciled_words:
        resolved_phonemes = build_phoneme_tier(reconciled_words)
    elif resolved_phonemes is None and alignment.aligned_chunks:
        all_aligned_words = [w for c in alignment.aligned_chunks for w in c.words]
        resolved_phonemes = build_phoneme_tier(all_aligned_words)

    phoneme_tier_items = (
        [
            {"start_sec": w.start_sec, "end_sec": w.end_sec, "text": w.word}
            for w in resolved_phonemes
            if w.word and w.word.strip()
        ]
        if resolved_phonemes
        else []
    )
    phoneme_intervals = build_contiguous_intervals(phoneme_tier_items, total_end)
    builder.create_tier("Phoneme", phoneme_intervals)

    # Optional Raw ASR Emissions tier
    raw_token_items = [
        {
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "text": t.word,
        }
        for t in (alignment.raw_tokens or [])
    ]
    if raw_token_items:
        raw_token_intervals = build_contiguous_intervals(raw_token_items, total_end)
        builder.create_tier("Raw ASR Emissions", raw_token_intervals)

    return builder.write(output_path)


class DialogueAlignmentPipeline:
    """
    End-to-end Code-Switched Dialogue Alignment Pipeline.

    Orchestrates:
    1. Multi-speaker transcript text parsing and speaker prefix extraction (e.g., 'Guy Soldier: ...').
    2. Code-switched token discrimination isolating English loanwords from Cherokee phonotactic syncope/intrusion.
    3. Continuous acoustic log-probability extraction via CherokeeASRModel / ModelOutput.
    4. Optional Silero VAD soft-masking.
    5. Syncope- and intrusion-aware CTC segmentation aligner.
    6. Syllabary-to-ASR phonetic reconciliation.
    7. Exporting canonical 7-tier Praat TextGrids and structured JSON manifests.
    """

    def __init__(
        self,
        asr_model: Optional[CherokeeASRModel] = None,
        aligner_config: Optional[CTCAlignerConfig] = None,
        ctc_aligner: Optional[Any] = None,
        projector: Optional[Any] = None,
        code_switched: bool = True,
        strip_speaker: bool = True,
    ) -> None:
        self.asr_model = asr_model
        self.config = aligner_config or CTCAlignerConfig()
        self.projector = projector or get_default_projector()
        self.preparer = CodeSwitchedPreparer(
            projector=self.projector,
            contextual_preaspiration=self.config.contextual_preaspiration,
        )
        self.code_switched = code_switched
        self.strip_speaker = strip_speaker

        if ctc_aligner is not None:
            self.ctc_aligner = ctc_aligner
        else:
            from transcription.alignment.ctc_aligner import (
                CTCSegmentationAligner as LegacyCTCSegmentationAligner,
            )

            self.ctc_aligner = LegacyCTCSegmentationAligner(
                model=self.asr_model,
                config=self.config,
            )

    def load_transcript(
        self,
        transcript: Union[
            str,
            Path,
            Sequence[Union[str, Dict[str, Any]]],
            Dict[str, Union[str, Dict[str, Any]]],
        ],
    ) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
        """
        Parses input transcript into normalized TextChunks and rich source metadata.
        """
        from transcription.alignment.ingestion import load_syllabary_transcript

        return load_syllabary_transcript(
            source=transcript,
            projector=self.projector,
            code_switched=self.code_switched,
            strip_speaker=self.strip_speaker,
            contextual_preaspiration=self.config.contextual_preaspiration,
        )

    def align(
        self,
        audio: Union[str, Path, AudioSegment, np.ndarray],
        transcript: Union[
            str,
            Path,
            Sequence[Union[str, Dict[str, Any]]],
            Dict[str, Union[str, Dict[str, Any]]],
        ],
        source_id: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> Tuple[AlignmentOutput, Dict[str, Dict[str, Any]]]:
        """
        Executes code-switched alignment of audio against transcript.
        """
        chunks, source_lookup = self.load_transcript(transcript)
        resolved_source_id = source_id or (
            str(audio) if isinstance(audio, (str, Path)) else "in_memory_dialogue"
        )
        effective_cache = self.config.cache if cache is None else bool(cache)

        alignment = self.ctc_aligner.align(
            audio_input=audio,
            chunks=chunks,
            source_id=resolved_source_id,
            asr_model=self.asr_model,
            cache=effective_cache,
            source_metadata=source_lookup,
        )

        return alignment, source_lookup

    def run(
        self,
        audio: Union[str, Path, AudioSegment, np.ndarray],
        transcript: Union[
            str,
            Path,
            Sequence[Union[str, Dict[str, Any]]],
            Dict[str, Union[str, Dict[str, Any]]],
        ],
        output_dir: Optional[Union[str, Path]] = None,
        reconcile: bool = True,
        export_praat: bool = True,
        export_manifest: bool = True,
        textgrid_filename: str = "dialogue_alignment.TextGrid",
        manifest_filename: str = "dialogue_manifest.json",
        cache: Optional[bool] = None,
    ) -> AlignmentOutput:
        """
        Full turnkey execution of dialogue alignment, reconciliation,
        7-tier Praat TextGrid generation, and manifest export.
        """
        alignment, source_lookup = self.align(
            audio=audio,
            transcript=transcript,
            cache=cache,
        )

        syllabary_lookup = {
            cid: meta.get("syllabary", meta.get("cherokee", meta.get("text", "")))
            for cid, meta in source_lookup.items()
        }
        syllabary_words_tier = build_syllabary_word_tier(alignment, syllabary_lookup)

        additional_word_tiers: Dict[str, Sequence[WordInterval]] = {
            "Syllabary Words": syllabary_words_tier,
        }

        english_words_tier: Optional[List[WordInterval]] = None
        if self.code_switched:
            english_words_tier = build_english_word_tier(alignment, source_lookup)
            additional_word_tiers["English Words"] = english_words_tier

        reconciled_words_tier: Optional[List[WordInterval]] = None
        if reconcile:
            reconciled_words_tier = reconcile_alignment_words(
                alignment, syllabary_lookup
            )
            additional_word_tiers["Reconciled Words"] = reconciled_words_tier

        phoneme_words_tier = build_phoneme_tier(
            reconciled_words_tier
            or [w for c in alignment.aligned_chunks for w in c.words]
        )
        additional_word_tiers["Phoneme"] = phoneme_words_tier

        if self.config and getattr(self.config, "enable_vad_soft_masking", False):
            vad_intervals = extract_vad_intervals(
                audio=audio,
                p_low=getattr(self.config, "vad_p_low", 0.15),
                p_high=getattr(self.config, "vad_p_high", 0.60),
                pad_ms=getattr(self.config, "vad_pad_ms", 60),
            )
            additional_word_tiers["VAD State"] = [
                WordInterval(word=label, start_sec=s, end_sec=e)
                for s, e, label in vad_intervals
            ]

        if output_dir is not None:
            out_dir_path = Path(output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)

            if export_manifest:
                export_manifest_file(
                    alignment=alignment,
                    output_dir=out_dir_path,
                    filename=manifest_filename,
                    source_metadata=source_lookup,
                    additional_word_tiers=additional_word_tiers,
                )

            if export_praat:
                export_7tier_textgrid(
                    alignment=alignment,
                    output_dir=out_dir_path,
                    filename=textgrid_filename,
                    source_metadata=source_lookup,
                    syllabary_words=syllabary_words_tier,
                    english_words=english_words_tier,
                    reconciled_words=reconciled_words_tier,
                    phoneme_words=phoneme_words_tier,
                )

        return alignment


def align_dialogue(
    audio: Union[str, Path, AudioSegment, np.ndarray],
    transcript: Union[
        str,
        Path,
        Sequence[Union[str, Dict[str, Any]]],
        Dict[str, Union[str, Dict[str, Any]]],
    ],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[CherokeeASRModel] = None,
    config: Optional[CTCAlignerConfig] = None,
    reconcile: bool = True,
    export_praat: bool = True,
    export_manifest: bool = True,
    textgrid_filename: str = "dialogue_alignment.TextGrid",
    manifest_filename: str = "dialogue_manifest.json",
    projector: Optional[Any] = None,
    code_switched: bool = True,
    strip_speaker: bool = True,
    cache: bool = True,
) -> AlignmentOutput:
    """
    Convenience turnkey procedure for code-switched dialogue alignment with 7-tier Praat export.
    """
    pipeline = DialogueAlignmentPipeline(
        asr_model=model,
        aligner_config=config,
        projector=projector,
        code_switched=code_switched,
        strip_speaker=strip_speaker,
    )
    return pipeline.run(
        audio=audio,
        transcript=transcript,
        output_dir=output_dir,
        reconcile=reconcile,
        export_praat=export_praat,
        export_manifest=export_manifest,
        textgrid_filename=textgrid_filename,
        manifest_filename=manifest_filename,
        cache=cache,
    )


def align_syllabary_ctc(
    audio: Union[str, Path, AudioSegment, np.ndarray],
    transcript: Union[str, Path, List[str], List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[CherokeeASRModel] = None,
    model_path: Optional[str] = None,
    config: Optional[CTCAlignerConfig] = None,
    cache: bool = True,
    reconcile: bool = True,
    export_praat: bool = True,
    export_manifest: bool = True,
    textgrid_filename: str = "ctc_alignment.TextGrid",
    manifest_filename: str = "alignment_manifest.json",
    projector: Optional[Any] = None,
    code_switched: bool = False,
    strip_speaker: bool = False,
    contextual_preaspiration: Optional[bool] = None,
) -> AlignmentOutput:
    """
    Turnkey syncope- and intrusion-aware CTC Segmentation runner for syllabary audio and transcripts.
    Maintains exact backwards compatibility with transcription.alignment.pipeline.align_syllabary_ctc.
    """
    effective_contextual_preaspiration = (
        contextual_preaspiration
        if contextual_preaspiration is not None
        else (config.contextual_preaspiration if config is not None else True)
    )
    cfg = config or CTCAlignerConfig(
        contextual_preaspiration=effective_contextual_preaspiration
    )
    asr_model = model
    if asr_model is None:
        token = os.environ.get("HF_TOKEN", None)
        asr_model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=model_path,
            token=token,
        )

    pipeline = DialogueAlignmentPipeline(
        asr_model=asr_model,
        aligner_config=cfg,
        projector=projector,
        code_switched=code_switched,
        strip_speaker=strip_speaker,
    )
    return pipeline.run(
        audio=audio,
        transcript=transcript,
        output_dir=output_dir,
        reconcile=reconcile,
        export_praat=export_praat,
        export_manifest=export_manifest,
        textgrid_filename=textgrid_filename,
        manifest_filename=manifest_filename,
        cache=cache,
    )


def align_syllabary_greedy(
    audio: Union[str, Path, AudioSegment, np.ndarray],
    transcript: Union[str, Path, List[str], List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[CherokeeASRModel] = None,
    model_path: Optional[str] = None,
    distance_metric: Optional[Any] = None,
    skip_vad: bool = False,
    reconcile: bool = True,
    export_praat: bool = True,
    export_manifest: bool = True,
    textgrid_filename: str = "greedy_alignment.TextGrid",
    manifest_filename: str = "alignment_manifest.json",
    emissions_extractor: Optional[Any] = None,
    projector: Optional[Any] = None,
    code_switched: bool = False,
    strip_speaker: bool = False,
) -> AlignmentOutput:
    """
    Aligns Cherokee audio against a syllabary transcript using the established
    Greedy ASR Inference + Sliding-Window DTW / Needleman-Wunsch pipeline with
    Syllabary Reconciliation.
    Maintains exact backwards compatibility with transcription.alignment.pipeline.align_syllabary_greedy.
    """
    from transcription.alignment.ingestion import load_syllabary_transcript

    chunks, source_lookup = load_syllabary_transcript(
        transcript,
        projector=projector,
        code_switched=code_switched,
        strip_speaker=strip_speaker,
    )

    from transcription.alignment.aligner import (
        NeedlemanWunschWordAligner,
        SlidingWindowDTWAligner,
    )
    from transcription.alignment.extractors import (
        ASREmissionsExtractor as _ASREmissionsExtractor,
        CherokeeASRExtractor as _CherokeeASRExtractor,
    )

    if emissions_extractor is not None:
        extractor = emissions_extractor
    else:
        asr_model = model
        if asr_model is None:
            token = os.environ.get("HF_TOKEN", None)
            asr_model = CherokeeASRModel.from_pretrained_or_best(
                path_or_repo=model_path,
                token=token,
            )
        extractor = _CherokeeASRExtractor(model=asr_model, skip_vad=skip_vad)

    audio_source_id = (
        str(audio) if isinstance(audio, (str, Path)) else "in_memory_audio"
    )
    emissions = extractor.extract(audio)

    word_aligner = NeedlemanWunschWordAligner(distance_metric=distance_metric)
    aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    alignment = aligner.align(
        emissions=emissions, chunks=chunks, source_id=audio_source_id
    )

    syllabary_lookup = {
        cid: meta.get("syllabary", meta.get("cherokee", meta.get("text", "")))
        for cid, meta in source_lookup.items()
    }
    syllabary_words_tier = build_syllabary_word_tier(alignment, syllabary_lookup)

    additional_word_tiers: Dict[str, Sequence[WordInterval]] = {
        "Syllabary Words": syllabary_words_tier,
    }

    english_words_tier: Optional[List[WordInterval]] = None
    if code_switched:
        english_words_tier = build_english_word_tier(alignment, source_lookup)
        additional_word_tiers["English Words"] = english_words_tier

    reconciled_words_tier: Optional[List[WordInterval]] = None
    if reconcile:
        reconciled_words_tier = reconcile_alignment_words(alignment, syllabary_lookup)
        additional_word_tiers["Reconciled Words"] = reconciled_words_tier

    phoneme_words_tier = build_phoneme_tier(
        reconciled_words_tier or [w for c in alignment.aligned_chunks for w in c.words]
    )
    additional_word_tiers["Phoneme"] = phoneme_words_tier

    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if export_manifest:
            export_manifest_file(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=manifest_filename,
                source_metadata=source_lookup,
                additional_word_tiers=additional_word_tiers,
            )

        if export_praat:
            export_7tier_textgrid(
                alignment=alignment,
                output_dir=out_dir_path,
                filename=textgrid_filename,
                source_metadata=source_lookup,
                syllabary_words=syllabary_words_tier,
                english_words=english_words_tier,
                reconciled_words=reconciled_words_tier,
                phoneme_words=phoneme_words_tier,
            )

    return alignment


__all__ = [
    "DialogueAlignmentPipeline",
    "align_dialogue",
    "align_syllabary_ctc",
    "align_syllabary_greedy",
    "build_english_word_tier",
    "build_phoneme_tier",
    "build_speaker_intervals",
    "build_syllabary_word_tier",
    "build_turn_intervals",
    "export_7tier_textgrid",
]
