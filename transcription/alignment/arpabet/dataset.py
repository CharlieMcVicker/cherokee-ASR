# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.dataset

Dataset extraction, phonetic balancing, audio waveform slicing, and manifest
generation for English speech corpora (LibriSpeech).

Produces 16kHz mono PCM .wav single-word clips padded by +/-25ms, balanced across
all 39 ARPAbet phonemes with frequency capping to eliminate stopword dominance,
and serialized as a durable words_manifest.json conforming to WordManifestEntry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio

from transcription.alignment.arpabet.forced_aligner import (
    AlignedWordSpan,
    MMSForcedAligner,
    get_default_forced_aligner,
)
from transcription.alignment.arpabet.g2p import (
    G2pExtractor,
    extract_arpabet,
    get_default_g2p,
)
from transcription.alignment.arpabet.types import (
    STANDARD_ARPABET_PHONEMES,
    ArpabetToken,
    WordManifestEntry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordCandidate:
    """
    Immutable candidate record for an aligned word extracted from an utterance.
    """

    candidate_id: str
    utterance_id: str
    word: str
    start_sec: float
    end_sec: float
    duration: float
    arpabet: Tuple[ArpabetToken, ...]
    audio_path: str = ""
    speaker_id: Optional[str] = None
    score: float = 0.0
    utterance_index: Optional[int] = None

    @property
    def arpabet_phones(self) -> Tuple[str, ...]:
        return tuple(tok.phone for tok in self.arpabet)


class PhoneticWordBalancer:
    """
    Balances word selection across all 39 ARPAbet phonemes while enforcing
    frequency capping (e.g. max 2 occurrences per unique word string).

    Supports both streaming selection and corpus-wide subset optimization.
    """

    def __init__(
        self,
        target_count: int = 5000,
        max_per_word: int = 2,
        phoneme_vocab: Sequence[str] = STANDARD_ARPABET_PHONEMES,
    ) -> None:
        self.target_count: int = target_count
        self.max_per_word: int = max_per_word
        self.phoneme_vocab: Tuple[str, ...] = tuple(phoneme_vocab)
        self.phoneme_vocab_set: Set[str] = set(self.phoneme_vocab)

        self.phoneme_counts: Dict[str, int] = {p: 0 for p in self.phoneme_vocab}
        self.word_counts: Dict[str, int] = {}
        self.selected: List[WordCandidate] = []

    def can_accept(self, word: str, arpabet: Sequence[ArpabetToken]) -> bool:
        """
        Check whether a word candidate is eligible for acceptance.
        Rejects candidates that exceed the word frequency cap or lack standard phonemes.
        """
        lemma = word.strip().upper()
        if not lemma:
            return False

        if self.word_counts.get(lemma, 0) >= self.max_per_word:
            return False

        if not arpabet:
            return False

        # Reject words containing non-standard phonemes
        for tok in arpabet:
            if tok.phone not in self.phoneme_vocab_set:
                return False

        return True

    def score_candidate(self, arpabet: Sequence[ArpabetToken]) -> float:
        """
        Compute marginal utility of accepting a candidate word.
        Words containing rare or underrepresented phonemes score significantly higher.
        """
        if not arpabet:
            return 0.0

        min_count = min(self.phoneme_counts.values()) if self.phoneme_counts else 0
        score = 0.0

        for tok in arpabet:
            p = tok.phone
            count = self.phoneme_counts.get(p, 0)
            score += 1.0 / ((1.0 + count) ** 2)
            if count == min_count:
                score += 10.0

        return score

    def add(self, candidate: WordCandidate) -> bool:
        """
        Add a candidate to the balanced set if eligible.
        Updates phoneme counts and word frequency records.
        """
        if not self.can_accept(candidate.word, candidate.arpabet):
            return False

        lemma = candidate.word.strip().upper()
        self.word_counts[lemma] = self.word_counts.get(lemma, 0) + 1

        for tok in candidate.arpabet:
            p = tok.phone
            if p in self.phoneme_counts:
                self.phoneme_counts[p] += 1

        self.selected.append(candidate)
        return True

    def select_balanced_subset(
        self,
        candidates: Sequence[WordCandidate],
        target_count: Optional[int] = None,
    ) -> List[WordCandidate]:
        """
        Select a phonetically balanced subset of words from a candidate pool.

        Applies max-min phoneme fairness and lemma frequency capping.
        Prioritizes underrepresented phonemes iteratively.
        """
        limit = target_count if target_count is not None else self.target_count
        if not candidates or limit <= 0:
            return []

        # Reset counts for fresh selection
        counts: Dict[str, int] = {p: 0 for p in self.phoneme_vocab}
        lemma_counts: Dict[str, int] = {}
        chosen: List[WordCandidate] = []
        chosen_ids: Set[str] = set()

        # Build candidate lookup by phoneme
        phone_to_cands: Dict[str, List[WordCandidate]] = {
            p: [] for p in self.phoneme_vocab
        }
        for c in candidates:
            lemma = c.word.strip().upper()
            if not lemma or not c.arpabet:
                continue
            if not all(t.phone in self.phoneme_vocab_set for t in c.arpabet):
                continue
            for t in c.arpabet:
                phone_to_cands[t.phone].append(c)

        # Index pointers for each phoneme list to avoid quadratic rescans
        phone_ptrs: Dict[str, int] = {p: 0 for p in self.phoneme_vocab}

        while len(chosen) < limit:
            # Find the phoneme with lowest count that still has eligible candidates
            sorted_phonemes = sorted(self.phoneme_vocab, key=lambda p: (counts[p], p))
            selected_cand: Optional[WordCandidate] = None

            for p in sorted_phonemes:
                cands = phone_to_cands[p]
                ptr = phone_ptrs[p]
                while ptr < len(cands):
                    cand = cands[ptr]
                    ptr += 1
                    if cand.candidate_id in chosen_ids:
                        continue
                    lemma = cand.word.strip().upper()
                    if lemma_counts.get(lemma, 0) >= self.max_per_word:
                        continue
                    selected_cand = cand
                    break
                phone_ptrs[p] = ptr
                if selected_cand is not None:
                    break

            if selected_cand is None:
                # No more candidates satisfy constraints via phone indexing
                # Try any remaining unselected candidate satisfying lemma cap
                for c in candidates:
                    if c.candidate_id in chosen_ids:
                        continue
                    lemma = c.word.strip().upper()
                    if lemma_counts.get(lemma, 0) >= self.max_per_word:
                        continue
                    if all(t.phone in self.phoneme_vocab_set for t in c.arpabet):
                        selected_cand = c
                        break

            if selected_cand is None:
                # All eligible candidates exhausted
                break

            # Accept candidate
            chosen.append(selected_cand)
            chosen_ids.add(selected_cand.candidate_id)
            lemma = selected_cand.word.strip().upper()
            lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
            for t in selected_cand.arpabet:
                counts[t.phone] += 1

        # Synchronize instance state
        self.phoneme_counts = counts
        self.word_counts = lemma_counts
        self.selected = chosen

        return chosen

    def get_summary(self) -> Dict[str, Any]:
        """
        Produce summary statistics of current balanced word distribution.
        """
        vals = list(self.phoneme_counts.values())
        min_p = (
            min(self.phoneme_counts.items(), key=lambda kv: kv[1])
            if self.phoneme_counts
            else ("", 0)
        )
        max_p = (
            max(self.phoneme_counts.items(), key=lambda kv: kv[1])
            if self.phoneme_counts
            else ("", 0)
        )
        mean_val = float(np.mean(vals)) if vals else 0.0
        std_val = float(np.std(vals)) if vals else 0.0
        active_phonemes = sum(1 for v in vals if v > 0)

        return {
            "total_words": len(self.selected),
            "unique_words": len(self.word_counts),
            "active_phonemes": active_phonemes,
            "total_phonemes": len(self.phoneme_vocab),
            "coverage_ratio": round(
                active_phonemes / max(1, len(self.phoneme_vocab)), 4
            ),
            "min_phoneme": {"phone": min_p[0], "count": min_p[1]},
            "max_phoneme": {"phone": max_p[0], "count": max_p[1]},
            "mean_phoneme_count": round(mean_val, 2),
            "std_phoneme_count": round(std_val, 2),
            "phoneme_counts": dict(self.phoneme_counts),
        }


def extract_word_clip(
    waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    output_path: Union[Path, str],
    padding_sec: float = 0.025,
    target_sample_rate: int = 16000,
) -> float:
    """
    Extract a single-word audio clip with boundary padding and save as 16kHz mono PCM .wav.

    Args:
        waveform: Audio waveform tensor [channels, samples] or [samples].
        sample_rate: Source audio sample rate in Hz.
        start_sec: Word start timestamp in seconds.
        end_sec: Word end timestamp in seconds.
        output_path: Target destination path for .wav file.
        padding_sec: Symmetric acoustic boundary padding in seconds (default 25ms).
        target_sample_rate: Standardized output sample rate (default 16000 Hz).

    Returns:
        Actual duration of the written audio clip in seconds.
    """
    # Normalize tensor shape to [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 2:
        waveform = waveform.view(waveform.shape[0], -1)

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    total_samples = waveform.shape[-1]
    total_sec = total_samples / target_sample_rate

    # Apply symmetric padding clamped to [0, total_sec]
    padded_start_sec = max(0.0, start_sec - padding_sec)
    padded_end_sec = min(total_sec, end_sec + padding_sec)

    start_idx = max(0, int(round(padded_start_sec * target_sample_rate)))
    end_idx = min(total_samples, int(round(padded_end_sec * target_sample_rate)))

    # Guarantee at least 1 sample if indices collapse
    if end_idx <= start_idx:
        end_idx = min(total_samples, start_idx + int(target_sample_rate * 0.05))

    clip_samples = waveform[:, start_idx:end_idx]
    clip_np = clip_samples.squeeze(0).cpu().numpy()

    # Save as 16-bit PCM WAV
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), clip_np, samplerate=target_sample_rate, subtype="PCM_16")

    clip_duration = (end_idx - start_idx) / target_sample_rate
    return round(clip_duration, 4)


def save_words_manifest(
    entries: Sequence[WordManifestEntry],
    manifest_path: Union[Path, str],
) -> None:
    """
    Save list of WordManifestEntry instances as durable JSON manifest.
    """
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.to_dict() for e in entries]
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_words_manifest(manifest_path: Union[Path, str]) -> List[WordManifestEntry]:
    """
    Load WordManifestEntry instances from durable JSON manifest.
    Supports both raw list shapes and wrapped dict structures.
    """
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [WordManifestEntry.from_dict(d) for d in data]
    elif isinstance(data, dict) and "entries" in data:
        return [WordManifestEntry.from_dict(d) for d in data["entries"]]
    raise ValueError(f"Unrecognized manifest format at {manifest_path}")


def collect_utterance_candidates(
    waveform: torch.Tensor,
    sample_rate: int,
    transcript: str,
    utterance_id: str,
    aligner: MMSForcedAligner,
    g2p: G2pExtractor,
    audio_path: str = "",
    speaker_id: Optional[str] = None,
    min_word_duration: float = 0.06,
    utterance_index: Optional[int] = None,
) -> List[WordCandidate]:
    """
    Align words in an utterance and generate WordCandidate records with stress-stripped ARPAbet tokens.
    """
    spans = aligner.align(waveform, sample_rate, transcript)
    if not spans:
        return []

    candidates: List[WordCandidate] = []
    for idx, span in enumerate(spans):
        # Filter out extremely brief or zero-duration glitches
        if span.duration < min_word_duration:
            continue

        raw_word = span.word.strip()
        arpabet_tokens = g2p.extract(raw_word, strip_stress=True)
        if not arpabet_tokens:
            continue

        cand_id = f"{utterance_id}_{idx:03d}"
        candidates.append(
            WordCandidate(
                candidate_id=cand_id,
                utterance_id=utterance_id,
                audio_path=audio_path,
                word=raw_word.upper(),
                start_sec=span.start_sec,
                end_sec=span.end_sec,
                duration=span.duration,
                arpabet=arpabet_tokens,
                speaker_id=speaker_id,
                score=span.score,
                utterance_index=utterance_index,
            )
        )

    return candidates


def iter_librispeech_utterances(
    librispeech_dir: Union[Path, str],
) -> Iterable[Tuple[str, Path, str, str]]:
    """
    Iterate over LibriSpeech utterances from disk, yielding (utterance_id, audio_path, transcript, speaker_id).
    """
    root = Path(librispeech_dir)
    # Search for dev-clean or root directory containing .trans.txt files
    trans_files = sorted(root.rglob("*.trans.txt"))
    for tf in trans_files:
        speaker_dir = tf.parent.parent.name
        lines = tf.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            audio_path = tf.parent / f"{utt_id}.flac"
            if not audio_path.exists():
                # Check for .wav alternative
                audio_path = tf.parent / f"{utt_id}.wav"
            if not audio_path.exists():
                continue
            speaker_id = utt_id.split("-")[0] if "-" in utt_id else speaker_dir
            yield utt_id, audio_path, transcript, speaker_id


def extract_balanced_dataset(
    librispeech_dir: Union[Path, str] = "data/librispeech",
    output_dir: Union[Path, str] = "data/arpabet_alignment",
    target_count: int = 5000,
    max_per_word: int = 2,
    padding_sec: float = 0.025,
    device: Optional[str] = None,
    max_utterances: Optional[int] = None,
) -> Tuple[List[WordManifestEntry], Dict[str, Any]]:
    """
    Extract phonetically balanced word clips and manifest from LibriSpeech corpus.

    1. Scans utterances and runs MMS_FA forced alignment.
    2. Extracts stress-stripped ARPAbet tokens using g2p_en.
    3. Runs PhoneticWordBalancer to select 5,000 words balanced across 39 ARPAbet phonemes.
    4. Slices audio waveforms with +/-25ms padding and saves 16kHz mono PCM .wav files.
    5. Writes durable words_manifest.json adhering to WordManifestEntry.

    Returns:
        Tuple of (list of WordManifestEntry, summary dictionary).
    """
    out_base = Path(output_dir)
    words_dir = out_base / "words"
    words_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_base / "words_manifest.json"

    aligner = MMSForcedAligner(device=device)
    g2p = get_default_g2p()
    balancer = PhoneticWordBalancer(
        target_count=target_count, max_per_word=max_per_word
    )

    logger.info("Scanning LibriSpeech utterances from %s...", librispeech_dir)
    utterances = list(iter_librispeech_utterances(librispeech_dir))
    if max_utterances is not None and max_utterances > 0:
        utterances = utterances[:max_utterances]

    logger.info("Found %d utterances to process", len(utterances))

    # Phase 1: Collect candidates
    all_candidates: List[WordCandidate] = []
    for idx, (utt_id, audio_path, transcript, spk_id) in enumerate(utterances):
        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
        except Exception as e:
            logger.warning("Failed to read audio %s: %s", audio_path, e)
            continue

        wf = torch.from_numpy(data)
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)
        elif wf.ndim > 1:
            wf = wf.mean(dim=0, keepdim=True)

        candidates = collect_utterance_candidates(
            waveform=wf,
            sample_rate=sr,
            transcript=transcript,
            utterance_id=utt_id,
            aligner=aligner,
            g2p=g2p,
            audio_path=str(audio_path),
            speaker_id=spk_id,
            utterance_index=idx,
        )
        all_candidates.extend(candidates)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(utterances):
            logger.info(
                "Processed %d/%d utterances, collected %d word candidates",
                idx + 1,
                len(utterances),
                len(all_candidates),
            )

    logger.info(
        "Finished candidate extraction. Total candidates: %d. Selecting %d balanced words...",
        len(all_candidates),
        target_count,
    )

    # Phase 2: Select balanced subset
    selected = balancer.select_balanced_subset(
        all_candidates, target_count=target_count
    )
    logger.info(
        "Selected %d balanced words across %d unique lemmas",
        len(selected),
        len(balancer.word_counts),
    )

    # Phase 3: Slice waveforms and export audio clips
    # Group candidates by audio_path to minimize disk I/O
    cands_by_audio: Dict[str, List[Tuple[int, WordCandidate]]] = {}
    for clip_idx, cand in enumerate(selected, start=1):
        cands_by_audio.setdefault(cand.audio_path, []).append((clip_idx, cand))

    entries: List[WordManifestEntry] = []
    for audio_path, cand_group in cands_by_audio.items():
        try:
            data, sr = sf.read(audio_path, dtype="float32")
            wf = torch.from_numpy(data)
            if wf.ndim == 1:
                wf = wf.unsqueeze(0)
            elif wf.ndim > 1:
                wf = wf.mean(dim=0, keepdim=True)
        except Exception as e:
            logger.error("Error reading source audio %s: %s", audio_path, e)
            continue

        for clip_num, cand in cand_group:
            clip_id = f"word_{clip_num:05d}"
            clip_filename = f"{clip_id}.wav"
            clip_path = words_dir / clip_filename
            relative_audio_path = f"data/arpabet_alignment/words/{clip_filename}"

            actual_duration = extract_word_clip(
                waveform=wf,
                sample_rate=sr,
                start_sec=cand.start_sec,
                end_sec=cand.end_sec,
                output_path=clip_path,
                padding_sec=padding_sec,
                target_sample_rate=16000,
            )

            entry = WordManifestEntry(
                clip_id=clip_id,
                audio_path=relative_audio_path,
                word=cand.word,
                duration=actual_duration,
                arpabet=cand.arpabet,
                start_sec=cand.start_sec,
                end_sec=cand.end_sec,
                speaker_id=cand.speaker_id,
            )
            entries.append(entry)

    # Sort entries by clip_id
    entries.sort(key=lambda e: e.clip_id)

    # Phase 4: Save manifest
    save_words_manifest(entries, manifest_path)
    summary = balancer.get_summary()
    summary["manifest_path"] = str(manifest_path)
    summary["words_dir"] = str(words_dir)
    summary["clips_extracted"] = len(entries)

    logger.info(
        "Manifest saved to %s with %d entries. Phoneme coverage: %.2f%%",
        manifest_path,
        len(entries),
        summary["coverage_ratio"] * 100,
    )

    return entries, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Extract phonetically balanced word clips and manifest from LibriSpeech"
    )
    parser.add_argument(
        "--librispeech-dir",
        type=str,
        default="data/librispeech",
        help="Root directory of LibriSpeech corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/arpabet_alignment",
        help="Target output directory for word clips and manifest",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5000,
        help="Target number of balanced word clips to extract",
    )
    parser.add_argument(
        "--max-per-word",
        type=int,
        default=2,
        help="Maximum occurrences per unique word string",
    )
    parser.add_argument(
        "--padding-sec",
        type=float,
        default=0.025,
        help="Boundary padding in seconds (+/-25ms)",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional cap on number of utterances to process",
    )
    args = parser.parse_args()

    entries, summary = extract_balanced_dataset(
        librispeech_dir=args.librispeech_dir,
        output_dir=args.output_dir,
        target_count=args.target_count,
        max_per_word=args.max_per_word,
        padding_sec=args.padding_sec,
        max_utterances=args.max_utterances,
    )

    print("\nExtraction complete!")
    print(f"Clips extracted: {len(entries)}")
    print(f"Unique words: {summary['unique_words']}")
    print(
        f"Active phonemes: {summary['active_phonemes']} / {summary['total_phonemes']}"
    )
    print(f"Manifest path: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
