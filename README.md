# Workshop Transcription & Wav2Vec2 Dataset Preparation

This repository contains tools for processing audio transcription files, segmenting raw recordings, labeling low-confidence predictions, and preparing/training speech recognition models like **Wav2Vec2** locally or in Docker containers.

## Repository Structure

```
workshop-transcription/
├── transcription/                # Main package and source code
│   ├── audio/                    # Audio preprocessing and segmentation
│   │   ├── segment.py            # Hyperparameter sweep / segment evaluation
│   │   └── extract.py            # Extract audio segments and write manifest
│   ├── inference/                # Model inference and active labeling
│   │   ├── single.py             # Run inference on a single audio file
│   │   ├── batch.py              # Batch inference on a directory of WAVs
│   │   ├── infer.py              # Core greedy inference and confidence scoring
│   │   └── labeler.py            # Web-based interface for low-confidence labeling
│   ├── timestamping/             # Ground-truth DTW timestamp alignment pipeline
│   │   ├── align_cli.py          # Unified CLI runner for timestamp alignment
│   │   ├── aligner.py            # Dynamic Time Warping (DTW) & Needleman-Wunsch aligner
│   │   ├── audio_segmenter.py    # VAD speech chunking for long audio
│   │   ├── exporter.py           # Praat TextGrid & JSON manifest export
│   │   └── prepare_ground_truth.py # Ingest Bible metadata or chunk JSON lists
│   ├── syllabary_enrichment/     # Phonetic rule merger & reconciliation engine
│   │   ├── alignment_engine.py   # Fine-grained character & syllable level alignment engine
│   │   ├── enrich_syllabary.py   # Core rule engine for vowel syncopation & aspiration transfer
│   │   ├── evaluate_reconciliation.py # Evaluation framework scoring Reconciled CER vs Raw ASR CER
│   │   └── inspect_pipeline.py   # Visual step-by-step diagnostic CLI for single recordings
│   ├── training/                 # Training, evaluation, and data prep
│   │   ├── prepare_csv.py        # Prepare training splits from raw CSV
│   │   ├── train.py              # Offline/local/remote Wav2Vec2 training
│   │   ├── evaluate_checkpoint.py # Score checkpoint against test set
│   │   └── evaluate_revisions.py  # Score Git revisions from Hugging Face
│   └── utils/                    # Utilities and checks
│       ├── syllabary_map.py      # Centralized Cherokee Syllabary to respelled phonetic mapping
│       ├── model_utils.py        # Checkpoint selection and config loading
│       ├── tone_normalization.py  # Normalizes tones in transcripts
│       └── verify_mps.py          # Verifies PyTorch MPS backend support
│
├── data/                         # Dedicated data directories
│   ├── raw/                      # Raw unsegmented audio files and datasets
│   ├── processed/                # Segmented audio folders and dataset CSV splits
│   └── results/                  # Inference outputs and evaluation results
│
├── timestamping_test_data/       # Test data & sample input generators for timestamping
│
├── archive/                      # Installer files and ZIP backups
│
└── Dockerfile                    # Container configuration file
```

---

## Workspace Setup

A virtual environment is managed locally via `uv` or standard Python `venv`.

1. **Activate virtual environment** (e.g. `.venv`).
2. **Install requirements**: Make sure you have `cmake` and `ffmpeg` installed on your system. Run:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Set your python path to the project root:
   ```bash
   export PYTHONPATH=".:${PYTHONPATH}"
   ```

---

## Packaged Entrypoints & Main Use Cases

All logic is package-based. Always run commands from the project root directory with `PYTHONPATH=.` (or after exporting it).

### 1. Audio Segmentation & Extraction

Analyze audio files to find speaking segments, or extract them into segmented WAV files.

- **Evaluate segmentation settings (Hyperparameter Sweep)**:
  ```bash
  python3 -m transcription.audio.segment data/raw/Bessie-Summerfield.wav --sweep
  ```
- **Extract segments to disk**:
  ```bash
  python3 -m transcription.audio.extract data/raw/Bessie-Summerfield.wav --out-dir data/processed/segments
  ```

### 2. Dataset Preparation

Split your local CSV dataset and WAV directory into Train, Validation, and Test partitions.

```bash
python3 -m transcription.training.prepare_csv \
  --csv data/processed/sentence_audio.csv \
  --audio-dir data/processed/sentence_audio \
  --output-prefix data/processed/cim-wav2vec2
```

### 3. Local Model Training

Train the Wav2Vec2 model offline.

```bash
python3 -m transcription.training.train \
  --train-csv data/processed/cim-wav2vec2-train.csv \
  --valid-csv data/processed/cim-wav2vec2-valid.csv \
  --test-csv data/processed/cim-wav2vec2-test.csv \
  --audio-dir data/processed/sentence_audio \
  --output-dir output_w2v2 \
  --epochs 50
```

### 4. Running Inference

- **Single file inference**:
  ```bash
  python3 -m transcription.inference.single data/raw/Bessie-Summerfield-2.wav \
    --checkpoint charliemcvicker/asr-cherokee
  ```
- **Batch inference on a directory**:
  ```bash
  python3 -m transcription.inference.batch data/processed/segments \
    --checkpoint charliemcvicker/asr-cherokee \
    --output data/results/batch_inference_results.csv
  ```

### 5. Web-based Active Labeler

Run a local labeling UI to manually review and label low-confidence audio segments:

```bash
python3 -m transcription.inference.labeler --port 8000
```

Then visit `http://localhost:8000/` in your browser. It automatically pulls data from `data/results/batch_inference_results.csv` and saves human-labeled transcripts to `data/processed/train_labeled.csv`.

### 6. Model Evaluation

- **Evaluate local checkpoint**:
  ```bash
  python3 -m transcription.training.evaluate_checkpoint \
    --test-csv data/processed/cim-wav2vec2-test.csv \
    --audio-dir data/processed/sentence_audio \
    --checkpoint remote_output_w2v2/checkpoint-800
  ```
- **Evaluate multiple Hugging Face commits/revisions**:
  ```bash
  python3 -m transcription.training.evaluate_revisions \
    --revisions-csv data/results/revisions_to_test.tsv \
    --test-csv data/processed/cim-wav2vec2-test.csv \
    --audio-dir data/processed/sentence_audio \
    --output-csv data/results/revision_scores.csv
  ```

### 7. Syllabary Enrichment Engine (Phonetic Reconciliation)

The Syllabary Enrichment engine reconciles ground-truth Cherokee Syllabary text against raw ASR acoustic emissions. It uses fine-grained character/syllable dynamic programming alignment while maintaining Cherokee Syllabary as the immutable structural anchor.

#### Core Reconciliation Principles
1. **Vowel Syncopation / Deletion**: Respects ASR when vowels are omitted/dropped in speech.
2. **Aspiration Transfer**: Systematically transfers pre-aspiration (`h-`), laryngeal/digraph aspiration (`th`, `kh`, `lh`, `nh`, `wh`, `yh`, `rh`, `sh`, `ch`), and post-vocalic aspiration (`-h`) emitted by ASR onto the base syllabary unit.
3. **Glottal Stop Filtering**: Preserves trailing glottal stops and onset glottal stops that accompany explicit onset consonants, while filtering standalone onset glottal stops (e.g. ASR `'a` for base `ya` $\rightarrow$ `ya`).

#### Evaluating Reconciliation CER vs Raw ASR CER

To calculate performance improvements (Reconciled CER vs Raw ASR CER) across train, validation, and test splits:

```bash
python3 -m transcription.syllabary_enrichment.evaluate_reconciliation \
  training_data/processed/split_audio_syl_target.csv \
  --force-recompute
```

#### Step-by-Step Diagnostic Inspection

To inspect the step-by-step alignment and rule actions for a specific recording (keyed by audio path or filename):

```bash
python3 -m transcription.syllabary_enrichment.inspect_pipeline \
  --record-id Sentence_for_entry_1136_01.wav
```

### 8. Inference Tooling - Ground-Truth Timestamp Alignment

The timestamping pipeline aligns ground-truth text (such as story transcripts or Bible verses) to long-form audio recordings. It uses VAD pre-segmentation, extracts CTC emissions from the ASR model, and applies Dynamic Time Warping (DTW) character/token alignment with Needleman-Wunsch fusion to calculate precise word start and end timestamps.

#### Usage Example

```bash
align-cherokee \
  --audio 'timestamping_test_data/Cherokee Story-Our Fishing Trip.wav' \
  --chunk-list 'timestamping_test_data/fishing_story.json' \
  --output-dir timestamping_test_data/fishing \
  --export-praat
```

#### Ground-Truth Input Formats

You can provide ground-truth transcript data using either `--chunk-list` or `--bible-metadata`:

1. **Chunk List JSON (`--chunk-list`)**: A JSON array of segment objects. You can generate this using `timestamping_test_data/make_json.py`.
   ```json
   [
     {
       "line_id": "segment_001",
       "raw_phonetic": "tsani ahwesolvtanvi",
       "cherokee_syllabary": "ᏣᏂ ᎠᏪᏐᎸᏔᏅᎢ"
     }
   ]
   ```
2. **Bible Metadata JSON (`--bible-metadata` or `--metadata`)**: A JSON dictionary mapping verse identifiers to ground-truth text strings.
   ```json
    "020101": {
      "image_path": "images/020101.png",
      "english": "The beginning of the gospel of Jesus Christ, the Son of God;",
      "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ, ᏥᏌ ᎦᎶᏁᏛ ᎤᏁᎳᏅᎯ ᎤᏪᏥ ᎤᏤᎵᎦ.",
      "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv, Tsi-sa Ga-lo-ne-dv U-ne-la-nv-hi U-we-tsi u-tse-li-ga."
    },
   ```

#### Optional CLI Arguments

- `--model-path`: Path to a custom local checkpoint directory or Hugging Face model repository (defaults to best local config or `facebook/wav2vec2-base-960h`).
- `--export-praat`: Exports a Praat `.TextGrid` file alongside the JSON manifest (enabled by default).

#### Generated Output Artifacts

Running the alignment pipeline writes the following files to `--output-dir`:

- **`alignment_manifest.json`**: Structured alignment output including:
  - Verse/segment text, start/end timestamps, and matched Cherokee Syllabary word mappings.
  - Aligned words with exact start and end times.
  - Alignment evaluation metrics (Matched GT Verses ratio, Matched-Verse CER, and Character counts).
- **`alignment.TextGrid`**: Praat TextGrid annotation file containing:
  - **Ground Truth Words**: GT words mapped onto aligned time intervals.
  - **Padded GT Words**: GT word boundaries padded slightly to avoid truncation.
  - **ASR Model Emissions**: Raw acoustic CTC emissions emitted by the Wav2Vec2 model.

---

## 9. Building Desktop Executables (PyInstaller)

The repository includes a desktop launcher (`syllabary_transcriber`) built with **PyWebView** and **FastAPI** bundled into standalone executables via **PyInstaller**.

### Building for Windows (on a Windows Machine)

> **Note**: PyInstaller cannot cross-compile. To generate a native Windows `.exe`, you must run these steps natively on a Windows machine or VM.

1. **Set up Conda Environment**:
   ```cmd
   conda create -n cherokee-asr python=3.11 -y
   conda activate cherokee-asr
   ```

2. **Install Python & Packaging Dependencies**:
   ```cmd
   pip install -r requirements.txt
   pip install pyinstaller
   ```

3. **Compile the React UI**:
   ```cmd
   cd syllabary_transcriber\ui
   npm install
   npm run build
   cd ..\..
   ```

4. **Run PyInstaller Build**:
   ```cmd
   set KMP_DUPLICATE_LIB_OK=TRUE
   pyinstaller syllabary_transcriber/packaging/app.spec --workpath transcriber_build --distpath transcriber_dist --noconfirm
   ```

5. **Locate Build Output**:
   The standalone Windows executable and bundled runtime will be created under:
   `transcriber_dist\Cherokee Syllabary Transcriber\Cherokee Syllabary Transcriber.exe`

### Building for macOS

On macOS, execute:
```bash
npm --prefix syllabary_transcriber/ui run build
KMP_DUPLICATE_LIB_OK=TRUE pyinstaller syllabary_transcriber/packaging/app.spec --workpath transcriber_build --distpath transcriber_dist --noconfirm
```
The output `.app` bundle will be generated at:
`transcriber_dist/Cherokee Syllabary Transcriber.app`

