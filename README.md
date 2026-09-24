# Cherokee Speech Recognition (ASR) & Transcription Toolkit

A modular Python toolkit for Cherokee speech recognition, dataset preparation, phonetic syllabary enrichment, audio segmentation, ground-truth timestamp alignment, and real-time desktop digohwelisgi.

---

## 📚 Documentation Index

Detailed documentation for each subsystem is organized in the [`docs/`](docs/) directory:

| Guide | Description | Key Modules / Tools |
|---|---|---|
| **[Ground-Truth Alignment](docs/alignment.md)** | Sliding-Window DTW & Needleman-Wunsch word DP alignment, Praat TextGrid & JSON manifest export. | `digohwelisgi.core.alignment`, `digohwelisgi.apps.cli`, `align-cherokee` |
| **[Models & Inference](docs/models_and_inference.md)** | `CherokeeASRModel` encapsulation, `ModelOutput` currency, tiered procedural inference, and Web Active Labeler. | `digohwelisgi.core.models`, `digohwelisgi.cherokee.models` |
| **[Syllabary Enrichment](docs/syllabary_enrichment.md)** | Character/syllable DP alignment and rule merger engine (syncopation, aspiration transfer, glottal filtering) with diagnostic inspector. | `digohwelisgi.cherokee.enrichment`, `digohwelisgi.pipelines.enrichment` |
| **[Audio Segmentation](docs/audio_segmentation.md)** | Hybrid VAD audio chunking (`segment_long_audio`), energy profiling, and Silero VAD soft-masking. | `digohwelisgi.core.audio` |
| **[Training & Evaluation](docs/training_and_evaluation.md)** | Multi-domain dataset preparation, offline/remote Wav2Vec2 training, checkpoint evaluation, and HF revision benchmarks. | `digohwelisgi.training`, `digohwelisgi.evaluation` |
| **[Desktop Transcriber App](docs/desktop_transcriber.md)** | Standalone desktop application (PyWebView + React TypeScript + FastAPI) and PyInstaller build guides for macOS and Windows. | `syllabary_transcriber` |

---

## 4-Tier Architecture

The repository is organized into four decoupled architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Applications & Entrypoints (digohwelisgi.apps)     │
│   • CLI (align-cherokee)                                    │
│   • Desktop Transcriber (syllabary_transcriber)             │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Domain Pipelines (digohwelisgi.pipelines)          │
│   • scripture: Continuous chapter alignment & verse slicing │
│   • dialogue: Code-switched interview alignment             │
│   • enrichment: Phonetic syllabary enrichment & alignment   │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Cherokee Domain (digohwelisgi.cherokee)            │
│   • orthography: Syllabary, DG, TTH conversion & tables     │
│   • phonotactics: Intrusions, syncope, surface constraints  │
│   • distance: Phonological confusion cost metrics           │
│   • codeswitching: Synthetic loanword target projection     │
│   • enrichment: Syllable reconciliation engine              │
│   • models: CherokeeASRModel factory & weights              │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Core Engine (digohwelisgi.core)                    │
│   • audio: Segmenting & Silero VAD soft-masking             │
│   • models: ModelOutput currency, inference & ASRModel      │
│   • alignment: DP (DTW, Needleman-Wunsch) & CTC trellis     │
│   • exporters: Praat TextGrid & Manifest serialization      │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
workshop-digohwelisgi/
├── docs/                         # Detailed modular technical documentation
│   ├── alignment.md              # Timestamping and DTW alignment guide
│   ├── audio_segmentation.md     # Audio preprocessing and VAD guide
│   ├── desktop_transcriber.md    # PyWebView / React desktop app guide
│   ├── models_and_inference.md   # CherokeeASRModel and inference guide
│   ├── syllabary_enrichment.md   # Syllabary reconciliation rule engine guide
│   └── training_and_evaluation.md# Wav2Vec2 training and evaluation guide
│
├── digohwelisgi/                # Core Python package (4-tier architecture)
│   ├── core/                     # Tier 1: Language-agnostic foundational engine
│   │   ├── alignment/            # DP (DTW, Needleman-Wunsch), CTC trellis, models, distance
│   │   ├── audio/                # AudioChunk, segment_long_audio, Silero VAD soft-masking
│   │   ├── exporters/            # Multi-tier Praat TextGrid & JSON manifest builders
│   │   └── models/               # ModelOutput universal currency, standalone inference, ASRModel
│   │
│   ├── cherokee/                 # Tier 2: Cherokee phonetic & linguistic domain logic
│   │   ├── codeswitching/        # SyntheticTargetProjector, CodeSwitchedPreparer, compound clitics
│   │   ├── distance/             # PhonologicalConfusionCostMetric, ConfusionMatrixCostMetric
│   │   ├── enrichment/           # SyllableAlignmentEngine, reconcile_phonetics
│   │   ├── models/               # CherokeeASRModel factory and weights loader
│   │   ├── orthography/          # Orthography enum, convert_orthography, syllabary tables, tones
│   │   └── phonotactics/         # Surface phonotactic rules, transition masks, prepare_cherokee_text
│   │
│   ├── pipelines/                # Tier 3: Domain use-case orchestration
│   │   ├── dialogue/             # DialogueAlignmentPipeline (code-switched interviews, 7-tier TextGrid)
│   │   ├── enrichment/           # EnrichmentPipeline (phonetic syllabary enrichment)
│   │   └── scripture/            # ScripturePipeline (continuous chapter alignment & verse slicing)
│   │
│   ├── apps/                     # Tier 4: Applications & CLI entrypoints
│   │   └── cli.py                # align-cherokee console script entrypoint
│   │
│   ├── evaluation/               # Model evaluation, confusion matrices, and manifold analysis
│   └── training/                 # Wav2Vec2 dataset preparation, training, and checkpoint benchmarking
│
├── syllabary_transcriber/        # Desktop transcription app (PyWebView + React + FastAPI)
│   ├── app.py                    # PyWebView desktop bridge & FastAPI server
│   ├── packaging/                # PyInstaller spec files
│   └── ui/                       # React 18 / TypeScript / Vite frontend
│
├── data/                         # Data directories (raw audio, processed splits, dictionaries)
├── scripts/                      # Declarative pipeline runner scripts (realign_bible.py, realign_gs_mm_ctc.py)
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

## Quick Usage Cheatsheet

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

### 2. Python Speech-to-Text Inference

Run inference programmatically using `CherokeeASRModel` and `ModelOutput`:

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

# Load model checkpoint
model = CherokeeASRModel.from_pretrained_or_best()

# Single audio inference
output = model.infer("data/raw/sample.wav")
print("Transcription:", output.decode_greedy())

# Batch inference
outputs = model.infer_batch(["sample1.wav", "sample2.wav"], batch_size=16)
for out in outputs:
    print("Batch Item:", out.decode_greedy())
```
*See [docs/models_and_inference.md](docs/models_and_inference.md) for full `ModelOutput` and inference details.*

### 3. Syllabary Enrichment

Reconcile native Cherokee Syllabary against acoustic ASR emissions:

```python
from digohwelisgi.pipelines.enrichment import align_and_enrich_syllabary

result = align_and_enrich_syllabary(
    audio="data/raw/sample.wav",
    syllabary="ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
)
print("Reconciled Phonetics:", result)
```
*See [docs/syllabary_enrichment.md](docs/syllabary_enrichment.md) for rule engine details.*

### 4. Continuous Scripture & Dialogue Realignment Scripts

Run declarative pipeline drivers:

```bash
# Realign New Testament chapters
python3 scripts/realign_bible.py --book mark --chapter 1

# Realign code-switched dialogue interview
python3 scripts/realign_gs_mm_ctc.py
```

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
pyright digohwelisgi
```
