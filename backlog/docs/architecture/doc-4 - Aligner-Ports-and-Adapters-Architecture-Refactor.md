---
id: doc-4
title: Aligner Ports and Adapters Architecture Refactor
type: specification
created_date: '2026-08-30 22:40'
updated_date: '2026-09-02 14:28'
---
# Alignment Engine Architecture & API Specification

## 1. Executive Summary & Design Principles

The `transcription.alignment` package implements an end-to-end, modular alignment and timestamping pipeline for Cherokee audio recordings and ground-truth text (story transcripts, Bible verses, or arbitrary JSON chunk lists).

### Core Architectural Principles:
1. **Streamlined Functional Architecture & Pure Dataclasses**:
   - The core domain exclusively processes generic chunk models: `TextChunk`, `TokenEmission`, `AlignedChunk`, `WordInterval`, `AlignmentMetrics`, and `AlignmentOutput`.
   - Core algorithms (sliding-window DTW, Needleman-Wunsch word DP alignment) are strictly agnostic to dataset-specific metadata schemas (such as Bible verses, chapter numbers, or line IDs).
2. **Pluggable Normalization & Distance Metrics**:
   - Text normalization for both reference text and acoustic ASR emissions is decoupled and injected as pure callable strategies (e.g. `normalize_text_for_alignment`).
   - Distance calculation is abstracted via `DistanceMetric` protocol implementations (`DefaultCERDistanceMetric`, `CharacterErrorRateMetric`, `LevenshteinDistanceMetric`, `CustomCallableDistanceMetric`).
3. **Decoupled Acoustic Extraction (`ASREmissionsExtractor`)**:
   - Audio emission extraction is abstracted via `ASREmissionsExtractor` protocol.
   - `CherokeeASRExtractor` interfaces with `CherokeeASRModel` and supports both automatic VAD segmentation and bypass mode (`skip_vad=True`).
4. **Pure Reconciliation Mapping**:
   - Phonetic reconciliation (`reconcile_word_intervals`, `reconcile_alignment_words`) enriches ground-truth Cherokee syllabary against acoustic emissions, producing immutable `WordInterval` instances with reconciled text.
5. **Multi-Tier Outbound Exporters**:
   - Outbound exporters generate standard JSON manifests (`alignment_manifest.json`), diagnostic debug JSON (`alignment_debug.json`), and multi-tier Praat TextGrids (`alignment.TextGrid`).

---

## 2. Package Structure & Module Organization

```
transcription/alignment/
├── __init__.py           # Unified exports for all models, aligners, extractors, metrics, and exporters
├── models.py             # Pure dataclasses (TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentMetrics, AlignmentOutput)
├── normalizers.py        # Phonetic & orthographic normalizer (normalize_text_for_alignment)
├── distance_metrics.py   # Distance metrics (DistanceMetric protocol, CER, Levenshtein)
├── extractors.py         # ASREmissionsExtractor protocol, CherokeeASRExtractor, Callback/Precomputed extractors
├── ingestion.py          # Ground-truth parsers (load_generic_chunks, load_bible_chunks)
├── aligner.py            # NeedlemanWunschWordAligner & SlidingWindowDTWAligner
├── reconciliation.py     # Syllabary/ASR phonetic reconciliation (reconcile_word_intervals, reconcile_alignment_words)
├── exporters.py          # Multi-tier Praat TextGrid, JSON manifest, and debug exporters
└── cli.py                # align-cherokee CLI entrypoint and run_alignment_pipeline orchestrator
```

---

## 3. Core Domain Models (`transcription.alignment.models`)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class TokenEmission:
    """An individual token or word emitted by the ASR model with timestamp bounds."""
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

@dataclass(frozen=True)
class TextChunk:
    """Generic text unit to be aligned against audio/emissions."""
    chunk_id: str
    text: str

@dataclass
class WordInterval:
    """Aligned word token with start/end bounds and emitted token details."""
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0
    flagged: bool = False
    emitted_word: Optional[str] = None

@dataclass
class AlignedChunk:
    """A matched segment with bounded timestamps and aligned words."""
    chunk_id: str
    start_sec: float
    end_sec: float
    words: List[WordInterval] = field(default_factory=list)
    distance_score: float = 1.0
    emitted_text: str = ""

@dataclass
class AlignmentMetrics:
    """Quality metrics across aligned chunks."""
    total_chunks: int
    matched_chunks: int
    match_ratio: float
    mean_distance_score: float
    total_ground_truth_chars: int
    total_emitted_chars: int

@dataclass
class AlignmentOutput:
    """Final output from alignment execution."""
    aligned_chunks: List[AlignedChunk]
    source_id: str = ""
    raw_tokens: List[TokenEmission] = field(default_factory=list)
    metrics: Optional[AlignmentMetrics] = None
```

---

## 4. Programmatic API & Usage Signatures

### 4.1 High-Level Orchestrator (`run_alignment_pipeline`)
```python
from transcription.alignment.cli import run_alignment_pipeline

alignment = run_alignment_pipeline(
    audio_path="data/raw/recording.wav",
    output_dir="data/results/alignment",
    chunk_list_path="data/processed/chunks.json",  # or bible_metadata_path="data/processed/verses.json"
    export_praat=True,
    export_manifest=True,
    model_path=None,   # Defaults to best config / HF checkpoint
    skip_vad=False,    # Set True if audio is pre-segmented
    debug_export=False,
    reconcile=True,    # Syllabary / ASR phonetic reconciliation
)
```

### 4.2 Low-Level Modular Pipeline
```python
from transcription.alignment import (
    CherokeeASRExtractor,
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
    export_manifest,
    export_textgrid,
    load_generic_chunks,
    normalize_text_for_alignment,
    reconcile_word_intervals,
)
from transcription.models.asr_model import CherokeeASRModel

# Ingestion
chunks, source_lookup = load_generic_chunks("data/processed/chunks.json")

# Extraction
model = CherokeeASRModel.from_pretrained("charliemcvicker/asr-cherokee")
extractor = CherokeeASRExtractor(model=model, skip_vad=False)
emissions = extractor.extract("data/raw/recording.wav")

# Alignment
word_aligner = NeedlemanWunschWordAligner(
    chunk_normalizer=normalize_text_for_alignment,
    emission_normalizer=normalize_text_for_alignment,
)
aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)
alignment = aligner.align(emissions=emissions, chunks=chunks, source_id="data/raw/recording.wav")

# Reconciliation & Export
export_manifest(alignment, output_dir="output", source_metadata=source_lookup)
export_textgrid(alignment, output_dir="output", source_metadata=source_lookup)
```
