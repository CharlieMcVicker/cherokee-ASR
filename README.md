# Cherokee Speech Recognition (ASR) & Transcription Toolkit

A modular Python toolkit for Cherokee speech recognition, dataset preparation, phonetic syllabary enrichment, audio segmentation, ground-truth timestamp alignment, and real-time desktop transcription.

---

## 📚 Documentation Index

Detailed documentation for each subsystem is organized in the [`docs/`](docs/) directory:

| Guide | Description | Key Modules / Tools |
|---|---|---|
| **[Ground-Truth Alignment](docs/alignment.md)** | Sliding-Window DTW & Needleman-Wunsch word DP alignment, Praat TextGrid & JSON manifest export. | `transcription.alignment`, `align-cherokee` |
| **[Models & Inference](docs/models_and_inference.md)** | `CherokeeASRModel` encapsulation, tiered inference primitives (`get_logits`, `get_word_confidences`), batch inference, and Web Active Labeler. | `transcription.models`, `transcription.inference` |
| **[Syllabary Enrichment](docs/syllabary_enrichment.md)** | Character/syllable DP alignment and rule merger engine (syncopation, aspiration transfer, glottal filtering) with diagnostic inspector. | `transcription.syllabary_enrichment` |
| **[Audio Segmentation](docs/audio_segmentation.md)** | Hybrid VAD audio chunking (`segment_long_audio`), energy profiling, and hyperparameter parameter sweeps. | `transcription.audio` |
| **[Training & Evaluation](docs/training_and_evaluation.md)** | Multi-domain dataset preparation, offline/remote Wav2Vec2 training, checkpoint evaluation, and HF revision benchmarks. | `transcription.training` |
| **[Desktop Transcriber App](docs/desktop_transcriber.md)** | Standalone desktop application (PyWebView + React TypeScript + FastAPI) and PyInstaller build guides for macOS and Windows. | `syllabary_transcriber` |

---

## Repository Structure

```
workshop-transcription/
├── docs/                         # Detailed modular technical documentation
│   ├── alignment.md              # Timestamping and DTW alignment guide
│   ├── audio_segmentation.md     # Audio preprocessing and VAD guide
│   ├── desktop_transcriber.md    # PyWebView / React desktop app guide
│   ├── models_and_inference.md   # CherokeeASRModel and inference guide
│   ├── syllabary_enrichment.md   # Syllabary reconciliation rule engine guide
│   └── training_and_evaluation.md# Wav2Vec2 training and evaluation guide
│
├── transcription/                # Core Python package
│   ├── alignment/                # Ground-truth DTW timestamp alignment pipeline
│   │   ├── aligner.py            # SlidingWindowDTWAligner & NeedlemanWunschWordAligner
│   │   ├── cli.py                # align-cherokee CLI runner & pipeline orchestrator
│   │   ├── distance_metrics.py   # Distance metrics (CER, Levenshtein, custom callable)
│   │   ├── exporters.py          # Praat TextGrid, JSON manifest, and debug exporters
│   │   ├── extractors.py         # ASREmissionsExtractor protocol & CherokeeASRExtractor
│   │   ├── ingestion.py          # Chunk ingestion parsers (generic & Bible JSON)
│   │   ├── models.py             # Domain models (TextChunk, TokenEmission, WordInterval, AlignedChunk)
│   │   ├── normalizers.py        # Phonetic & orthographic normalizers
│   │   └── reconciliation.py     # Pure syllabary phonetic word interval reconciliation
│   ├── audio/                    # Audio preprocessing and segmentation
│   ├── inference/                # Model inference, batch runners, and active labeler
│   ├── models/                   # CherokeeASRModel class and procedural inference
│   ├── syllabary_enrichment/     # Syllabary phonetic rule merger & evaluation
│   ├── training/                 # Dataset preparation, training, and evaluation
│   └── utils/                    # Syllabary maps, model configs, and helpers
│
├── syllabary_transcriber/        # Desktop transcription app (PyWebView + React + FastAPI)
│   ├── app.py                    # PyWebView desktop bridge & FastAPI server
│   ├── packaging/                # PyInstaller spec files
│   └── ui/                       # React 18 / TypeScript / Vite frontend
│
├── data/                         # Data directories (raw audio, processed splits, results)
├── scripts/                      # Standalone data processing & generation scripts
└── timestamping_test_data/       # Test audio and sample ground truth inputs
```

---

## Workspace Setup

### 1. Conda Environment Setup

```bash
# Create and activate environment
conda create -n cherokee-asr python=3.11 -y
conda activate cherokee-asr

# Install project dependencies
pip install -e ".[dev]"
```

Ensure `ffmpeg` and `cmake` are installed on your system.

### 2. Set Python Path

Always run commands from the project root directory:

```bash
export PYTHONPATH=".:${PYTHONPATH}"
```

---

## Quick CLI Cheatsheet

### 1. Timestamp Alignment (`align-cherokee`)

Align ground-truth transcripts to long-form audio with Praat `.TextGrid` export:

```bash
align-cherokee \
  --audio 'timestamping_test_data/Cherokee Story-Our Fishing Trip.wav' \
  --chunk-list 'timestamping_test_data/fishing_story.json' \
  --output-dir 'output/fishing' \
  --export-praat \
  --reconcile
```
*See [docs/alignment.md](docs/alignment.md) for full options and Python API examples.*

### 2. Speech-to-Text Inference

Run inference on a single audio file or batch directory:

```bash
# Single file inference
python3 -m transcription.inference.single data/raw/sample.wav \
  --checkpoint charliemcvicker/asr-cherokee

# Batch directory inference
python3 -m transcription.inference.batch data/processed/segments \
  --checkpoint charliemcvicker/asr-cherokee \
  --output data/results/batch_results.csv
```
*See [docs/models_and_inference.md](docs/models_and_inference.md) for `CherokeeASRModel` Python usage.*

### 3. Syllabary Enrichment & Diagnostic Inspection

Inspect character/syllable alignment and phonetic rule decisions for a recording:

```bash
python3 -m transcription.syllabary_enrichment.inspect_pipeline \
  --record-id Sentence_for_entry_1136_01.wav
```
*See [docs/syllabary_enrichment.md](docs/syllabary_enrichment.md) for rule engine details.*

### 4. Audio Segmentation

Evaluate segmentation parameters or extract segmented clips:

```bash
python3 -m transcription.audio.segment data/raw/recording.wav --sweep
python3 -m transcription.audio.extract data/raw/recording.wav --out-dir data/processed/segments
```
*See [docs/audio_segmentation.md](docs/audio_segmentation.md) for VAD parameter tuning.*

### 5. Desktop Transcriber App

Launch the Cherokee Syllabary desktop transcriber:

```bash
python3 -m syllabary_transcriber
```
*See [docs/desktop_transcriber.md](docs/desktop_transcriber.md) for PyInstaller packaging guides.*

---

## Testing & Code Quality

Run the test suite and static type checker:

```bash
pytest
pyright transcription
```

