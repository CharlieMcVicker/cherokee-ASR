# Cherokee Syllabary Transcriber (Desktop Application)

The **Cherokee Syllabary Transcriber** is a real-time, hands-free desktop dictation application designed for transcribing spoken Cherokee speech into authentic **Cherokee Syllabary** (ᏣᎳᎩ ᏗᎪᏪᎵ).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Application Architecture](#2-application-architecture)
   - [Technology Stack](#technology-stack)
   - [Audio Pipeline & Web Audio API JS VAD](#audio-pipeline--web-audio-api-js-vad)
   - [Frontend-Backend Communication Bridge](#frontend-backend-communication-bridge)
   - [Backend Inference & Thread Isolation](#backend-inference--thread-isolation)
   - [Cherokee Syllabary Transliteration](#cherokee-syllabary-transliteration)
3. [Running in Development Mode](#3-running-in-development-mode)
   - [Prerequisites](#prerequisites)
   - [Option A: Full Desktop PyWebView Window](#option-a-full-desktop-pywebview-window)
   - [Option B: Vite Dev Server + FastAPI Server](#option-b-vite-dev-server--fastapi-server)
4. [Building Standalone Executables (PyInstaller)](#4-building-standalone-executables-pyinstaller)
   - [Spec File Architecture (`app.spec`)](#spec-file-architecture-appspec)
   - [Building for macOS](#building-for-macos)
   - [Building for Windows](#building-for-windows)
   - [Troubleshooting Common Packaging Issues](#troubleshooting-common-packaging-issues)
5. [UI Features & Controls](#5-ui-features--controls)
   - [Microphone Setup & Device Enumeration](#microphone-setup--device-enumeration)
   - [Live Document Canvas](#live-document-canvas)
   - [Status Banner & Real-Time Indicators](#status-banner--real-time-indicators)
   - [Low-Confidence Audio Filter & Retry Prompt](#low-confidence-audio-filter--retry-prompt)
   - [Document Action Controls](#document-action-controls)

---

## 1. Overview

The desktop transcriber operates as a continuous, hands-free dictation tool. As speakers pronounce Cherokee words and phrases, a client-side Web Audio Voice Activity Detection (VAD) worklet segments active speech bursts, streams PCM audio buffers into the fine-tuned Wav2Vec2 CTC acoustic model, transliterates phonetic CTC tokens into Cherokee Syllabary, and appends the text directly to the interactive document canvas.

```
 [Microphone] ──> [AudioWorklet VAD] ──> [PyWebView API / REST] ──> [CherokeeASRModel]
                                                                            │
                                                                            ▼
 [Document Canvas] <── [React UI] <── [Syllabary Map Engine] <── [CTC Logits / Tokens]
```

---

## 2. Application Architecture

### Technology Stack

| Layer | Technologies | Role |
|---|---|---|
| **Desktop Shell** | [PyWebView](https://pywebview.flowrl.com/) | Native OS webview window hosting the React UI with direct Python JavaScript bindings. |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) | Embedded asynchronous ASGI web server providing REST endpoints and static asset hosting. |
| **Frontend UI** | [React 18](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/) | Modern responsive UI, audio stream management, and document state. |
| **Audio Processing** | Web Audio API (`AudioWorkletProcessor`) | Low-latency in-browser voice activity detection and 16kHz float32 PCM frame accumulation. |
| **ASR Inference** | [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/) | Fine-tuned `Wav2Vec2ForCTC` acoustic model wrapped by `CherokeeASRModel`. |
| **Transliteration** | `digohwelisgi.cherokee.orthography` | Phonetic-to-Syllabary deterministic conversion engine. |
| **Packaging** | [PyInstaller](https://pyinstaller.org/) | Bundles Python runtime, PyTorch dependencies, React build, and native assets into a single executable. |

---

### Audio Pipeline & Web Audio API JS VAD

Voice Activity Detection (VAD) is implemented using a custom client-side audio worklet (`syllabary_transcriber/ui/public/vad-processor.js`). Running in an isolated browser audio-rendering thread ensures zero latency and prevents UI stutter.

```
       Microphone Audio (16kHz Mono Stream)
                         │
                         ▼
           ┌───────────────────────────┐
           │ AudioWorkletProcessor     │ (vad-processor.js)
           │ - Computes RMS Energy     │
           │ - Zero-Crossing Analysis  │
           └─────────────┬─────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  [Silence Frame]                 [Active Frame]
  - Add to 200ms pre-roll buffer   - Count speech frames
  - Check redemption silence       - Buffer 16kHz PCM samples
         │                               │
         ▼                               ▼
  [Redemption Expired]            [Min Speech Met]
  - Trigger SPEECH_END            - Trigger SPEECH_START
  - Emit Float32Array PCM         - Pre-roll audio flushed
```

#### VAD Tuning Parameters
- **`energyThreshold` (`0.015`)**: Root-Mean-Square (RMS) amplitude threshold required to trigger voice detection.
- **`minSpeechFrames` (`3` frames $\approx 60\text{ms}$)**: Consecutive active frames required to signal speech onset and filter transient noises (clicks, mic bumps).
- **`redemptionFrames` (`35` frames $\approx 700\text{ms}$)**: Trailing silence frames required to signal speech completion, allowing natural micro-pauses between syllables without cutting off phrases.
- **`preRollFrames` (`10` frames $\approx 200\text{ms}$)**: Ring buffer preserving audio immediately prior to speech detection, ensuring initial consonants, glottal stops, and voiceless onsets (`h-`, `s-`) are not clipped.
- **Minimum Duration Check**: Rejects audio clips with fewer than $1,600\text{ samples}$ ($< 0.10\text{s}$) to avoid false triggers on background clicks.

---

### Frontend-Backend Communication Bridge

The frontend communicates with the Python backend through a unified bridge implemented in `syllabary_transcriber/ui/src/hooks/usePyWebView.ts`:

1. **Native PyWebView Mode**: When running inside the packaged desktop application, the hook directly invokes the injected JS API:
   ```javascript
   window.pywebview.api.transcribe_pcm(pcmSamples, 16000);
   ```
2. **Web / Dev Server Fallback**: When running inside a standalone browser during development, the hook falls back to a standard HTTP POST request:
   ```javascript
   fetch('/api/transcribe-pcm', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ pcmData: pcmSamples, sampleRate: 16000 }),
   });
   ```

---

### Backend Inference & Thread Isolation

The backend server is implemented in `syllabary_transcriber/app.py`:

```python
class SyllabaryApi:
    def __init__(self, model_dir: Optional[str] = None):
        self.asr_model: Optional[CherokeeASRModel] = None
        self.model_dir = model_dir

    def transcribe_pcm(self, pcm_data: Union[List[float], str], sample_rate: int = 16000) -> dict:
        self._ensure_model_loaded()
        ...
        result = self.asr_model.transcribe(audio_input=pcm_array, sample_rate=sample_rate)
        return result.to_dict()
```

#### Environment Isolation for GUI Stability
Desktop GUI applications running embedded PyTorch can encounter OpenMP conflicts or thread deadlocks with webview GUI event loops. `syllabary_transcriber/__main__.py` enforces strict CPU single-threading and environment isolation at launch:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # Enforce CPU execution for desktop stability
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Prevent OpenMP multiple runtime abort
os.environ["OMP_NUM_THREADS"] = "1"           # Restrict OpenMP to single-threaded CPU
os.environ["MKL_NUM_THREADS"] = "1"           # Restrict Intel MKL to single-threaded CPU
```

---

### Cherokee Syllabary Transliteration

When `CherokeeASRModel` emits phonetic tokens (e.g. `tsalagi`), the system translates them into Cherokee Syllabary via `digohwelisgi/utils/syllabary_map.py`:

- **Syllable Splitting**: Splits words into consonant-vowel combinations matching Cherokee vowels (`a`, `e`, `i`, `o`, `u`, `v`).
- **Aspiration & De-aspiration**: Translates aspirated series (e.g., `thv` $\rightarrow$ `tv` $\rightarrow$ **Ꮫ**, `khv` $\rightarrow$ `kv` $\rightarrow$ **Ꭼ**).
- **S-Clusters**: Handles initial `s-` consonant clusters (e.g., `hska` $\rightarrow$ `s` + `ka` $\rightarrow$ **Ꮝ** + **Ꭶ** $\rightarrow$ **ᏍᎦ**).
- **Pre-Aspiration & Liquids**: Supports `lh` $\rightarrow$ **Ꮭ** and `hna` $\rightarrow$ **Ꮏ**.

---

## 3. Running in Development Mode

### Prerequisites

1. **Python Environment**:
   - Activate the project conda environment:
     ```bash
     conda activate cherokee-asr
     ```
2. **Node.js Environment**:
   - Node.js version 18+ and npm installed.

---

### Option A: Full Desktop PyWebView Window

To launch the native desktop PyWebView window locally:

```bash
python3 -m syllabary_transcriber
```

**How it works**:
1. Checks whether `syllabary_transcriber/ui/dist` exists.
2. If missing, automatically compiles the React UI (`npm run build`).
3. Starts an embedded Uvicorn server in a background daemon thread on port `8765`.
4. Spawns the native OS PyWebView window ($960 \times 720\text{px}$).

---

### Option B: Vite Dev Server + FastAPI Server

For frontend UI development with hot-module reloading (HMR):

#### Terminal 1: Run FastAPI Dev Backend
```bash
python3 -m syllabary_transcriber --dev --port 8000
```
*Runs the Uvicorn server on `http://127.0.0.1:8000` with auto-reloading.*

#### Terminal 2: Run Vite Dev Server
```bash
cd syllabary_transcriber/ui
npm install
npm run dev
```
*Open `http://localhost:5173` in your browser. Audio calls will proxy to the FastAPI backend.*

---

## 4. Building Standalone Executables (PyInstaller)

PyInstaller packages the Python interpreter, PyTorch runtime, model weights, and compiled React frontend into a self-contained binary.

> [!IMPORTANT]
> PyInstaller does not support cross-compilation. To build a Windows `.exe`, you must run the build on a Windows machine. To build a macOS `.app`, you must run the build on macOS.

---

### Spec File Architecture (`app.spec`)

The build specification is located at `syllabary_transcriber/packaging/app.spec`:

1. **Static UI Ingestion**: Bundles `syllabary_transcriber/ui/dist` into `syllabary_transcriber/ui/dist` inside the distribution folder.
2. **Dynamic C-Library Hooks**: Uses `collect_all('torchvision')` and `collect_all('torchaudio')` to bundle shared libraries and C++ operator registrations (preventing runtime missing operator errors like `torchvision::nms`).
3. **Hidden Imports**: Explicitly imports `uvicorn`, `pywebview`, and ASGI drivers.
4. **macOS Microphone Entitlements**: Embeds the `NSMicrophoneUsageDescription` key in the macOS `Info.plist` bundle:
   ```python
   info_plist={
       'NSMicrophoneUsageDescription': 'Cherokee Syllabary Transcriber needs access to your microphone to transcribe speech.',
   }
   ```

---

### Building for macOS

Execute the following commands from the repository root:

```bash
# 1. Compile the React UI
npm --prefix syllabary_transcriber/ui run build

# 2. Run PyInstaller
KMP_DUPLICATE_LIB_OK=TRUE pyinstaller syllabary_transcriber/packaging/app.spec \
  --workpath transcriber_build \
  --distpath transcriber_dist \
  --noconfirm
```

#### Output Location:
```
transcriber_dist/Cherokee Syllabary Transcriber.app
```

You can drag this `.app` bundle into `/Applications` or double-click to launch.

---

### Building for Windows

Follow these steps on a Windows 10 or 11 (x64) machine:

#### 1. Setup Conda Environment
Open **Anaconda Prompt** or **PowerShell**:
```cmd
conda create -n cherokee-asr python=3.11 -y
conda activate cherokee-asr
```

#### 2. Install Dependencies
```cmd
pip install -r requirements.txt
pip install pyinstaller
```

#### 3. Compile the React UI
```cmd
cd syllabary_transcriber\ui
npm install
npm run build
cd ..\..
```

#### 4. Run PyInstaller
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
pyinstaller syllabary_transcriber/packaging/app.spec --workpath transcriber_build --distpath transcriber_dist --noconfirm
```

#### Output Location:
```
transcriber_dist\Cherokee Syllabary Transcriber\Cherokee Syllabary Transcriber.exe
```

---

### Troubleshooting Common Packaging Issues

| Issue | Cause | Solution |
|---|---|---|
| **`OMP: Error #15: Initializing libiomp5md.dll...`** | Multiple copies of OpenMP runtime bundled by PyTorch and TorchVision. | Ensure `set KMP_DUPLICATE_LIB_OK=TRUE` is exported before building and running. |
| **`RuntimeError: operator torchvision::nms does not exist`** | PyInstaller stripped torchvision binary operators. | `packaging/app.spec` uses `collect_all('torchvision')` to ensure all shared `.so` / `.dll` files are preserved. |
| **Microphone Permission Denied (macOS)** | macOS blocked audio input because `NSMicrophoneUsageDescription` was missing. | `packaging/app.spec` includes `NSMicrophoneUsageDescription` in the `BUNDLE` definition. |
| **`FileNotFoundError: ui/dist/index.html`** | React frontend was not compiled prior to running PyInstaller. | Run `npm run build` in `syllabary_transcriber/ui/` before packaging. |

---

## 5. UI Features & Controls

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  Cherokee Syllabary Transcriber                   Microphone: [Default ▾]  │
 ├────────────────────────────────────────────────────────────────────────────┤
 │  🟢 LISTENING — Speak a word syllable-by-syllable...             [ Pause ]  │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   ᎣᏏᏲ ᏂᎦᏓ! ᎪᎯ ᎢᎦ ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ᏕᏥᎪᎵᏰᎠ...                                  │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │  [ 📋 Copy to clipboard ]   [ ⌫ Delete last word ]   [ 🗑️ Clear all ]      │
 └────────────────────────────────────────────────────────────────────────────┘
```

### Microphone Setup & Device Enumeration

When launching the application for the first time:
- The app checks for microphone permissions via `navigator.mediaDevices.getUserMedia`.
- Audio input devices are dynamically enumerated using `navigator.mediaDevices.enumerateDevices()`.
- Users can select their preferred microphone (USB mic, headset, built-in array) from the dropdown selector.

### Live Document Canvas

- **Real-Time Display**: Displays accumulated Cherokee Syllabary text formatted in standard Unicode Syllabary characters.
- **Auto-Scrolling**: Automatically scrolls the viewport to the bottom as new transcribed words arrive.
- **Manual Editing**: Fully editable textarea allowing manual corrections, pasting, and spacing adjustments.

### Status Banner & Real-Time Indicators

The banner dynamically reflects the transcription state:
- 🟢 **LISTENING — Speak a word syllable-by-syllable...**: Active and waiting for speech.
- 🟡 **LISTENING — User speaking...**: Speech detected by VAD worklet.
- 🟣 **LISTENING — Transcribing audio...**: PCM audio buffer being processed by the ASR model.
- ⚪ **PAUSED**: Listening is paused via the **Pause** button.

### Low-Confidence Audio Filter & Retry Prompt

To prevent ambient noise or garbled speech from inserting erroneous characters into the document, `App.tsx` inspects the model's confidence score:

```typescript
if (typeof result.confidence === 'number') {
  const confValue = result.confidence <= 1.0 ? result.confidence * 100 : result.confidence;
  if (confValue < 90) {
    triggerFeedback('⚠️ Sorry, please say that again');
    return; // Discards low-confidence segment
  }
}
```

If average confidence falls below **90%**, the audio segment is rejected and a temporary toast notification (`⚠️ Sorry, please say that again`) is displayed in the status bar.

### Document Action Controls

The footer (`CommandFooter.tsx`) provides three rapid action buttons:

1. **📋 Copy to clipboard**: Copies the complete document contents to the system clipboard and displays a temporary `✓ Copied` confirmation.
2. **⌫ Delete last word**: Removes the most recent whitespace-delimited Cherokee word from the canvas.
3. **🗑️ Clear all**: Empties the document canvas.
