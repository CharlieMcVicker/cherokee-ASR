---
id: doc-11
title: Models and Inference Subsystem
type: guide
created_date: '2026-09-24 17:30'
updated_date: '2026-09-24 17:30'
---
# Models & Inference Guide

This guide provides an in-depth reference for model loading, procedural inference primitives, command-line interfaces, and active learning workflows in the Cherokee Speech-to-Text (`workshop-transcription`) system.

---

## Table of Contents

1. [Overview](#1-overview)
2. [CherokeeASRModel Class](#2-cherokeeasrmodel-class)
   - [Model Architecture & Encapsulation](#model-architecture--encapsulation)
   - [Factory Methods & Checkpoint Resolution](#factory-methods--checkpoint-resolution)
   - [Hardware Acceleration & Device Fallbacks](#hardware-acceleration--device-fallbacks)
   - [Model Caching](#model-caching)
3. [Procedural Inference Primitives](#3-procedural-inference-primitives)
   - [Data Structures: WordConfidence & ASRResult](#data-structures-wordconfidence--asrresult)
   - [Inference Pipeline Layers](#inference-pipeline-layers)
   - [Layer 1: Preprocessing & Logits (`get_logits`)](#layer-1-preprocessing--logits-get_logits)
   - [Layer 2: Softmax Probabilities (`get_probabilities`)](#layer-2-softmax-probabilities-get_probabilities)
   - [Layer 3: Word & Character Confidences (`get_word_confidences`)](#layer-3-word--character-confidences-get_word_confidences)
   - [Layer 4: Greedy CTC Decoding (`decode`)](#layer-4-greedy-ctc-decoding-decode)
   - [End-to-End Transcription (`transcribe` & `transcribe_batch`)](#end-to-end-transcription-transcribe--transcribe_batch)
4. [Model Inference & Execution](#4-model-inference--execution)
   - [Single Audio Inference (`CherokeeASRModel.infer`)](#single-audio-inference-cherokeeasrmodelinfer)
   - [High-Performance Batch Inference (`CherokeeASRModel.infer_batch`)](#high-performance-batch-inference-cherokeeasrmodelinfer_batch)
5. [Web-Based Active Labeler](#5-web-based-active-labeler)
   - [Workflow & Architecture](#workflow--architecture)
   - [Running the Labeler Server](#running-the-labeler-server)
   - [Active Learning Lifecycle](#active-learning-lifecycle)
6. [Programmatic Python Examples](#6-programmatic-python-examples)
   - [Basic Transcription](#basic-transcription)
   - [In-Memory Raw PCM Transcription](#in-memory-raw-pcm-transcription)
   - [Accessing Frame-Level & Word-Level Confidences](#accessing-frame-level--word-level-confidences)
   - [Custom Batch Processing](#custom-batch-processing)

---

## 1. Overview

The Cherokee ASR system is built on fine-tuned [Wav2Vec 2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2) Connectionist Temporal Classification (CTC) acoustic models. The core acoustic pipeline maps continuous 16kHz audio waveforms into phonetic transcriptions. Downstream modules provide transliteration into authentic **Cherokee Syllabary** (ᏣᎳᎩ ᏗᎪᏪᎵ) and phonetic rule reconciliation.

```
   Raw Audio (.wav / .mp3 / PCM buffer)
                    │
                    ▼
       ┌─────────────────────────┐
       │ Audio Preprocessing     │ (Resample to 16kHz mono, float32)
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Wav2Vec2 Acoustic Model │ (PyTorch CTC Forward Pass)
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Logits / Probabilities  │ [T, Vocab Size]
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Greedy CTC Decoder      │ (Argmax, deduplication, PAD removal)
       └────────────┬────────────┘
                    │
                    ├──────────────────────────┐
                    ▼                          ▼
       ┌─────────────────────────┐  ┌─────────────────────────┐
       │ Phonetic ASRResult      │  │ Word & Char Confidences │
       │ (e.g. "tsalagi")        │  │ (Timestamps, Top-5 Alts)│
       └────────────┬────────────┘  └─────────────────────────┘
                    │
                    ▼ (Downstream Syllabary Enrichment)
       ┌─────────────────────────┐
       │ Syllabary Transliteration│ (phonetics_to_syllabary / syllabary_enrichment)
       │ (e.g. "ᏣᎳᎩ")            │
       └─────────────────────────┘
```

The system is encapsulated in `CherokeeASRModel` (`digohwelisgi.cherokee.models.loader`), providing a tiered, decoupled architecture where each layer can be invoked independently or as part of a high-level end-to-end pipeline. The core model and `ASRResult` abstractions remain strictly language-agnostic (producing phonetic hypotheses), leaving Cherokee syllabary conversion and reconciliation to Tier 2 Cherokee domain modules (`digohwelisgi.cherokee.enrichment` and `digohwelisgi.cherokee.orthography`).

---

## 2. CherokeeASRModel Class

`CherokeeASRModel` is the primary model wrapper encapsulating `transformers.Wav2Vec2ForCTC` and `transformers.Wav2Vec2Processor`.

### Model Architecture & Encapsulation

```python
class CherokeeASRModel:
    def __init__(
        self,
        model: Wav2Vec2ForCTC,
        processor: Wav2Vec2Processor,
        device: Union[str, torch.device] = "cpu",
    )
```

The model instance holds references to:
- `self.model`: The underlying Hugging Face `Wav2Vec2ForCTC` neural network.
- `self.processor`: The `Wav2Vec2Processor` managing feature extraction and vocabulary tokenization.
- `self.device`: The active computation device (`"cuda"`, `"mps"`, or `"cpu"`).

### Factory Methods & Checkpoint Resolution

`CherokeeASRModel` provides four factory methods for instantiation:

#### 1. `CherokeeASRModel.from_pretrained`
Loads an explicit model checkpoint from a Hugging Face Hub repository ID or local filesystem path.

```python
@classmethod
def from_pretrained(
    cls,
    path_or_repo: str,
    revision: Optional[str] = None,
    processor_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    eval_mode: bool = True,
    use_cache: bool = True,
) -> CherokeeASRModel
```

**Parameters**:
- `path_or_repo` (`str`): Hugging Face repository ID (e.g., `"charliemcvicker/asr-cherokee"`) or directory path to saved model weights.
- `revision` (`Optional[str]`): Specific git branch, tag, or commit hash on Hugging Face Hub.
- `processor_path` (`Optional[str]`): Separate path for processor if different from model path (defaults to `path_or_repo`).
- `device` (`Optional[Union[str, torch.device]]`): Target PyTorch device. If `None`, automatically resolves to CUDA $\rightarrow$ MPS $\rightarrow$ CPU.
- `token` (`Optional[str]`): Hugging Face authentication token (falls back to `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` environment variables).
- `eval_mode` (`bool`): Automatically sets `model.eval()` (default: `True`).
- `use_cache` (`bool`): Caches model instances in memory across repeated calls (default: `True`).

#### 2. `CherokeeASRModel.from_config`
Loads a model using a dictionary specifying repository parameters.

```python
@classmethod
def from_config(
    cls,
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    eval_mode: bool = True,
    use_cache: bool = True,
) -> CherokeeASRModel
```

**Config Format**:
```json
{
  "repo": "charliemcvicker/asr-cherokee",
  "revision": "5464d15",
  "processor_path": null
}
```

#### 3. `CherokeeASRModel.get_best_model`
Automatically locates and loads the best-performing model defined in the repository's `best_model.json`.

```python
@classmethod
def get_best_model(
    cls,
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    eval_mode: bool = True,
    use_cache: bool = True,
) -> CherokeeASRModel
```

The underlying loader (`get_best_model_config` in `digohwelisgi/utils/model_utils.py`) searches upward through directory hierarchies to find `best_model.json`. If no file is found, it falls back to default checkpoint configuration: `{"repo": "charliemcvicker/asr-cherokee", "revision": "5464d15"}`.

#### 4. `CherokeeASRModel.from_pretrained_or_best`
Unified factory method that resolves checkpoints dynamically and provides automatic graceful fallback to public baseline models (default `facebook/wav2vec2-base-960h`).

```python
@classmethod
def from_pretrained_or_best(
    cls,
    path_or_repo: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    fallback_repo: str = "facebook/wav2vec2-base-960h",
    eval_mode: bool = True,
    use_cache: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    processor_path: Optional[str] = None,
) -> CherokeeASRModel
```

- If `path_or_repo` is provided, attempts to load via `from_pretrained`. If loading fails with an Exception, logs a warning and loads `fallback_repo`.
- If `path_or_repo` is `None`, attempts to load the best model configuration via `get_best_model`. If loading fails, logs a warning and loads `fallback_repo`.

### Hardware Acceleration & Device Fallbacks

The model resolution pipeline dynamically selects the fastest available hardware backend:

1. **NVIDIA CUDA**: Selected when `torch.cuda.is_available() == True`.
2. **Apple Silicon (MPS)**: Selected on macOS when `torch.backends.mps.is_available() == True`.
3. **CPU**: Selected when no GPU acceleration is available.

#### MPS Fallback Mechanism
Certain PyTorch CTC operations or edge tensor operations may raise a `NotImplementedError` on Apple Metal (MPS). `CherokeeASRModel` intercepts `NotImplementedError` during forward passes and automatically falls back to the CPU backend seamlessly without crashing the caller:

```python
try:
    with torch.no_grad():
        logits = self.model(input_tensor).logits
except NotImplementedError:
    if self.device == "mps":
        self.device = "cpu"
        self.model.to("cpu")
        input_tensor = input_tensor.to("cpu")
        with torch.no_grad():
            logits = self.model(input_tensor).logits
```

### Model Caching

To prevent redundant weight loading and VRAM bloat, `model_utils.py` maintains an in-memory cache `_MODEL_CACHE` indexed by `(path_or_repo, revision, processor_path, device_str, eval_mode)`. Successive calls to `get_model` or `CherokeeASRModel.from_pretrained` with matching parameters return the cached instance immediately.

---

## 3. Procedural Inference Primitives

`CherokeeASRModel` exposes a procedural API organized into 4 distinct layers:

```
[Audio Input] ──> Layer 1: get_logits() ──> Layer 2: get_probabilities() ──> Layer 3: get_word_confidences()
                                                                       │
                                                                       └──> Layer 4: decode() ──> ASRResult
```

### Data Structures: WordConfidence & ASRResult

All structured outputs are strongly typed dataclasses defined in `digohwelisgi.cherokee.models.loader`.

#### `WordConfidence`
Encapsulates word-level alignment, start/end timestamps, confidence score, and per-character breakdown.

```python
@dataclass
class WordConfidence:
    word: str                          # Phonetic word string (e.g. "osiyo")
    confidence: float                  # Mean token confidence (0.0 to 1.0)
    start_time: float                  # Start time in seconds (0.02s resolution)
    end_time: float                    # End time in seconds
    chars: List[Dict[str, Any]]        # Character-level details and alternative predictions

    def to_dict(self) -> Dict[str, Any]: ...
```

**Per-character dictionary structure in `chars`**:
```python
{
    "char": "s",
    "confidence": 0.9842,
    "start_time": 0.34,
    "alternatives": [
        {"char": "ts", "confidence": 0.0125},
        {"char": " ", "confidence": 0.0018}
    ]
}
```

#### `ASRResult`
Encapsulates complete acoustic model transcription output and word breakdown.

```python
@dataclass
class ASRResult:
    text: str                          # Decoded phonetic transcript
    transcription: str                 # Alias for text
    confidence: float                  # Average non-PAD token confidence (0.0 to 1.0)
    words: List[WordConfidence]        # Detailed word list (empty if compute_word_confidences=False)

    def to_dict(self) -> Dict[str, Any]: ...
    def __getitem__(self, item: str) -> Any: ...
    def get(self, item: str, default: Any = None) -> Any: ...
```

> **Note**: `ASRResult` implements `__getitem__` and `.get()`, enabling backwards-compatible dictionary-style indexing (`result["transcription"]`, `result["confidence"]`) as well as attribute access (`result.transcription`).

> **Language-Agnostic Abstraction**: `ASRResult` purposefully does not contain language-specific fields such as `syllabary`. The acoustic CTC decoder produces phonetic text hypotheses. Transliteration to Cherokee Syllabary and phonetic reconciliation are explicitly performed downstream by `digohwelisgi.cherokee.orthography.convert_orthography` or `digohwelisgi.cherokee.enrichment`.

---

### Inference Pipeline Layers

#### Layer 1: Preprocessing & Logits (`get_logits`)

```python
def get_logits(
    self,
    pcm_audio: Union[str, bytes, List[float], np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
) -> torch.Tensor
```

**Audio Preprocessing Helper (`CherokeeASRModel.preprocess_audio`)**:
- Accepts file paths (`.wav`, `.mp3`, `.flac`), raw byte buffers, Python float lists, NumPy arrays, or PyTorch tensors.
- Downmixes multi-channel audio to mono (`waveform.mean(dim=0)`).
- Resamples audio to 16,000 Hz using `torchaudio.transforms.Resample`.
- Returns a 1D `np.float32` array.

**Output**: PyTorch `torch.Tensor` of unnormalized logits with shape `[sequence_length, vocab_size]`.

---

#### Layer 2: Softmax Probabilities (`get_probabilities`)

```python
def get_probabilities(
    self,
    logits_or_audio: Union[torch.Tensor, np.ndarray, str, bytes, List[float]],
    sample_rate: int = 16000,
) -> torch.Tensor
```

Applies the softmax activation function across the vocabulary dimension. If raw audio is supplied instead of logits, `get_logits` is automatically called first.

**Output**: PyTorch `torch.Tensor` with shape `[sequence_length, vocab_size]` where values sum to 1.0 along the last dimension.

---

#### Layer 3: Word & Character Confidences (`get_word_confidences`)

```python
def get_word_confidences(
    self,
    probs_or_logits: Union[torch.Tensor, np.ndarray],
) -> List[WordConfidence]
```

Analyzes CTC emissions over time to build timestamped words:
1. Detects whether input contains raw logits or softmax probabilities.
2. Identifies non-PAD, non-duplicate character transitions.
3. Computes frame timestamps at $0.02\text{s}$ ($20\text{ms}$) per Wav2Vec2 CTC frame.
4. Identifies top-5 alternative character candidates per frame (`top_k_indices`).
5. Groups characters by word delimiter token (`" "`) into `WordConfidence` objects.

---

#### Layer 4: Greedy CTC Decoding (`decode`)

```python
def decode(
    self,
    logits_or_probs: Union[torch.Tensor, np.ndarray],
    compute_word_confidences: bool = True,
) -> ASRResult
```

Performs greedy CTC decoding on 2D or 3D logit/probability tensors:
1. Calculates $\text{argmax}$ across the vocabulary dimension for each time frame.
2. Decodes token IDs to phonetic text via `processor.batch_decode` or `processor.decode`.
3. Calculates sequence confidence by averaging probabilities over non-PAD tokens.
4. Returns an `ASRResult` containing decoded phonetic text and sequence confidence (keeping core acoustic decoding language-agnostic).
5. Optionally attaches word-level timestamp structures (`WordConfidence`).

---

#### End-to-End Transcription (`transcribe` & `transcribe_batch`)

##### Single Audio Transcription
```python
def transcribe(
    self,
    audio_input: Union[str, bytes, List[float], np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    compute_word_confidences: bool = True,
) -> ASRResult
```
Executes the full pipeline: `audio_input` $\rightarrow$ `preprocess_audio` $\rightarrow$ `get_logits` $\rightarrow$ `decode` $\rightarrow$ `ASRResult`.

##### Batched Audio Transcription
```python
def transcribe_batch(
    self,
    audio_inputs: Sequence[Union[str, bytes, List[float], np.ndarray, torch.Tensor]],
    sample_rate: int = 16000,
    batch_size: int = 16,
    compute_word_confidences: bool = False,
) -> List[ASRResult]
```

**Batch Optimization Features**:
- **Padding & Attention Masks**: Automatically constructs PyTorch attention masks to pad variable-length audio inputs.
- **Dynamic Length Unpadding**: Slices logits according to `model._get_feat_extract_output_lengths(input_lengths)` to prevent decoding padded silence.
- **OOM / cuDNN Recovery**: Automatically intercepts CUDA Out-Of-Memory (`torch.cuda.OutOfMemoryError`) or cuDNN engine failures, clears VRAM cache, and falls back to sequential single-item execution for the affected batch.
- **VRAM Defragmentation**: Calls `torch.cuda.empty_cache()` / `torch.mps.empty_cache()` after each batch.

---

## 4. Model Inference & Execution

The system provides streamlined procedural inference and unified container models (`ASRModel` and `CherokeeASRModel`).

### Single Audio Inference (`CherokeeASRModel.infer`)

Transcribes a single audio file and returns the universal `ModelOutput` currency containing logits, vocab mapping, and greedy decode projections.

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

model = CherokeeASRModel.from_pretrained_or_best()
output = model.infer("data/raw/Bessie-Summerfield-2.wav")

print("Transcription:", output.decode_greedy())
```

---

### High-Performance Batch Inference (`CherokeeASRModel.infer_batch`)

Processes multiple audio files with batched tensor execution:

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

model = CherokeeASRModel.from_pretrained_or_best()
audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]
outputs = model.infer_batch(audio_paths, batch_size=16)

for path, out in zip(audio_paths, outputs):
    print(f"{path}: {out.decode_greedy()}")
```
- `greedy_confidence`: Sequence confidence formatted to 4 decimal places (`0.0000` to `1.0000`).
- `word_confidences`: JSON string containing serialized `WordConfidence` objects with timestamps and alternatives.

---

## 5. Web-Based Active Labeler

`digohwelisgi/inference/labeler.py` provides a lightweight, interactive web application for human-in-the-loop active learning.

```
   ┌─────────────────────────────────────────┐
   │ Batch Inference Results CSV             │
   │ (data/results/batch_inference_results.csv)
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ Active Labeler Server (Port 8000)       │
   │ (Sorted by lowest confidence first)     │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ Reviewer Web UI (Listen, Edit, Enter)   │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ Curated Training CSV                    │
   │ (data/processed/train_labeled.csv)      │
   └─────────────────────────────────────────┘
```

### Workflow & Architecture

1. **Confidence Sorting**: Automatically sorts all unverified segments in ascending order of confidence. Segments with lowest model confidence (uncertain predictions) appear at the top.
2. **Visual Confidence Badges**:
   - 🔴 **Low (< 0.80)**: Urgent review required.
   - 🟡 **Medium (0.80 - 0.95)**: Likely minor phonetic errors.
   - 🟢 **High (> 0.95)**: High-accuracy predictions.
3. **Audio Playback**: Integrates native HTML5 audio playback (`<audio controls autoplay>`).
4. **Keyboard Shortcuts**: Pressing `Enter` inside the text box saves the label locally and immediately advances to the next lowest-confidence segment.
5. **State Preservation**: On server restart, reads any existing `data/processed/train_labeled.csv` to preserve previously labeled items.

### Running the Labeler Server

```bash
python3 -m digohwelisgi.inference.labeler --port 8000
```

Open `http://localhost:8000` in your web browser.

### Active Learning Lifecycle

1. Run batch inference across uncurated audio recordings using `CherokeeASRModel.infer_batch()`:
   ```python
   from digohwelisgi.cherokee.models.loader import CherokeeASRModel

   model = CherokeeASRModel.from_pretrained_or_best()
   # Process recordings to produce batch_inference_results.csv
   ```
2. Start the labeling server:
   ```bash
   python3 -m digohwelisgi.inference.labeler --port 8000
   ```
3. In the UI, review and correct the low-confidence entries. Click **Save CSV** to write out `data/processed/train_labeled.csv`.
4. Incorporate newly labeled samples into training splits:
   ```bash
   python3 -m digohwelisgi.training.prepare_csv \
     --csv data/processed/train_labeled.csv \
     --audio-dir data/raw/new_interviews \
     --output-prefix data/processed/active_learning_split
   ```

---

## 6. Programmatic Python Examples

### Basic Transcription

Transcribing an audio file to phonetic text and performing optional downstream Cherokee Syllabary transliteration:

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel
from digohwelisgi.cherokee.orthography import convert_orthography, Orthography

# 1. Load the recommended model checkpoint
asr = CherokeeASRModel.get_best_model()

# 2. Transcribe audio file (returns language-agnostic ASRResult)
result = asr.transcribe("data/raw/sample.wav")

# 3. Downstream Cherokee Syllabary transliteration
syllabary_text = convert_orthography(result.transcription, target=Orthography.SYLLABARY)

print("Phonetic Transcript :", result.transcription)
print("Cherokee Syllabary  :", syllabary_text)
print(f"Overall Confidence  : {result.confidence:.2%}")
```

---

### In-Memory Raw PCM Transcription

Transcribing live PCM audio arrays or raw bytes directly in memory (e.g., from a microphone or WebSocket stream):

```python
import numpy as np
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

asr = CherokeeASRModel.get_best_model()

# Example: 1-second 16kHz float32 audio buffer
pcm_samples = np.random.uniform(-0.1, 0.1, size=16000).astype(np.float32)

# Transcribe array directly to ASRResult
result = asr.transcribe(pcm_samples, sample_rate=16000)
print("Transcription:", result.transcription)
```

---

### Accessing Frame-Level & Word-Level Confidences

Extracting word boundaries, timestamps, and character-level alternatives:

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

asr = CherokeeASRModel.get_best_model()

# Obtain logits directly
logits = asr.get_logits("data/raw/sample.wav")

# Compute word and character breakdowns
words = asr.get_word_confidences(logits)

for w in words:
    print(f"Word: {w.word:<12} Time: {w.start_time:.2f}s - {w.end_time:.2f}s | Confidence: {w.confidence:.2%}")
    for c in w.chars:
        alts = ", ".join([f"{a['char']} ({a['confidence']:.2f})" for a in c["alternatives"][:2]])
        print(f"   Char '{c['char']}' @ {c['start_time']:.2f}s (conf: {c['confidence']:.2f}) -> Alts: {alts}")
```

---

### Custom Batch Processing

Performing batched inference over a list of file paths with custom batch sizing:

```python
from digohwelisgi.cherokee.models.loader import CherokeeASRModel

asr = CherokeeASRModel.get_best_model(device="cuda")

audio_files = [
    "data/raw/clip_01.wav",
    "data/raw/clip_02.wav",
    "data/raw/clip_03.wav",
]

# Run batched inference
results = asr.transcribe_batch(
    audio_inputs=audio_files,
    batch_size=16,
    compute_word_confidences=False
)

for path, res in zip(audio_files, results):
    print(f"[{path}] -> {res.transcription} (conf: {res.confidence:.4f})")
```
