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
│   ├── training/                 # Training, evaluation, and data prep
│   │   ├── prepare_csv.py        # Prepare training splits from raw CSV
│   │   ├── train.py              # Offline/local/remote Wav2Vec2 training
│   │   ├── evaluate_checkpoint.py # Score checkpoint against test set
│   │   └── evaluate_revisions.py  # Score Git revisions from Hugging Face
│   └── utils/                    # Utilities and checks
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

### 7. Inference Tooling - Ground-Truth Timestamp Alignment

The timestamping pipeline aligns ground-truth text (such as story transcripts or Bible verses) to long-form audio recordings. It uses VAD pre-segmentation, extracts CTC emissions from the ASR model, and applies Dynamic Time Warping (DTW) character/token alignment with Needleman-Wunsch fusion to calculate precise word start and end timestamps.

#### Usage Example

```bash
python3 -m transcription.timestamping.align_cli \
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