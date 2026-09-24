---
id: doc-23
title: 'Spec: Syllabary Transcriber Desktop App'
type: specification
created_date: '2026-09-24 17:30'
updated_date: '2026-09-24 17:30'
---
# Project Specification: Cherokee Syllabary Hands-Free Transcriber

## 1. Executive Summary

This project aims to build a lightweight, cross-platform desktop application designed specifically for Cherokee typists and language workers who have limited or no hand mobility. The application leverages a specialized syllable-based Cherokee Automatic Speech Recognition (ASR) model paired with in-browser Voice Activity Detection (VAD) to deliver a seamless, hands-free transcription experience.

---

## 2. Problem Statement & Core Objectives

* **Problem:** Existing speech-to-text workflows rely heavily on manual mouse clicks (e.g., "Start/Stop Recording" buttons) or standard keyboard push-to-talk triggers. For hands-free users, this introduces severe physical friction, fatigue, and speed bottlenecks.
* **Objective:** Enable hands-free input by using continuous silence detection (VAD) to slice audio word-by-word, feed it to a local ASR model, and output Cherokee Syllabary characters into a clean, document-style editor.

---

## 3. Key Target User Profile

* **Users:** Native speakers, Cherokees, and language workers generating Cherokee language documents.
* **Accessibility Context:** Non-hand-use primary mode. The UI must operate effectively without requiring active mouse or keyboard navigation during typing sessions, while remaining accessible to assistants or editors for manual tweaks.

---

## 4. System Architecture & Tech Stack

```
+-------------------------------------------------------------------------+
| Desktop Shell: pywebview (Python 3.10+ Native Engine Integration)       |
+-------------------------------------------------------------------------+
| Frontend Engine (HTML5 / JS / WASM)                                    |
|  • WebAssembly VAD (@ricky0123/vad-web / Silero VAD)                   |
|  • High-Legibility Document Textarea (Noto Sans / Plantagenet Cherokee)  |
+-------------------------------------------------------------------------+
                  |  (Float32Array PCM Audio via JS Bridge)
                  v
+-------------------------------------------------------------------------+
| Backend Engine (Python Embedded Runtime)                                |
|  • Syllable-based Cherokee ASR Inference Engine (PyTorch / ONNX)        |
|  • Local Web Server / IPC Bridge                                        |
|  • Cross-Platform Executable Packaging (PyInstaller / Nuitka)           |
+-------------------------------------------------------------------------+

```

### Component Breakdown

1. **Frontend UI (`pywebview` Container):** Uses host native web engines (Edge WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux). Runs local HTTP context (`[http://127.0.0.1](http://127.0.0.1)`) to ensure WebAssembly (WASM) compatibility.
2. **Audio Processing & VAD (Client-Side JS):**
* Employs `@ricky0123/vad-web` (Silero VAD running via WebAssembly).
* Detects continuous speech, triggers on silence thresholds (~600–800ms), cuts Float32 PCM audio buffers, and passes them to Python.


3. **ASR Model Execution (Python Backend):**
* Accepts audio buffers from the JS bridge.
* Runs local inference using the syllable-based Cherokee ASR model.
* Returns formatted Cherokee Syllabary strings (e.g., `ᎣᏏᏲ`) back to the frontend to append to the editor buffer.



---

## 5. User Interface & User Experience (UX)

### Design Philosophy

* **Word-Processor Clean:** A distraction-free, high-contrast, document-style interface (similar to Microsoft Word or Notepad).
* **High Legibility:** Default Cherokee font size set to **32px+** using Unicode-compliant Cherokee fonts (`Plantagenet Cherokee`, `Noto Sans Cherokee`).

### Wireframe Diagram

```
+-----------------------------------------------------------------------------------------+
| Cherokee Syllabary Transcriber                                  [ Mic: Built-in Mic  ▼ ] |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ 🟢 LISTENING ]  Say a word syllable-by-syllables. Pause when done.   [ ⏸ Pause Mic ] |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  ᎣᏏᏲ ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ...                                                                       |
|                                                                                         |
|                                                                                         |
|                                                                                         |
|                                                                                         |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
| Spoken Commands: "Delete" (remove last word) | "Clear All" (reset text)                |
| Status: Auto-copied to clipboard after each word                                        |
+-----------------------------------------------------------------------------------------+

```

### Complete HTML & CSS Frontend Reference Code

```html
<!DOCTYPE html>
<html lang="chr">
<head>
  <meta charset="UTF-8">
  <title>Cherokee Transcriber</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background-color: #f4f4f6;
      color: #111;
      display: flex;
      flex-direction: column;
      height: 100vh;
      padding: 20px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
    }
    h1 { font-size: 18px; font-weight: 600; color: #333; }
    select { padding: 4px 8px; font-size: 14px; }

    .status-banner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 15px;
      margin-bottom: 15px;
      background-color: #e6f4ea;
      color: #137333;
      border: 1px solid #ceead6;
      transition: all 0.2s ease;
    }

    .status-left {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
    }

    .status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background-color: #1e8e3e;
    }

    .status-banner.paused {
      background-color: #f1f3f4;
      color: #5f6368;
      border-color: #dadce0;
    }

    .btn-toggle {
      padding: 6px 14px;
      font-size: 14px;
      font-weight: 600;
      border-radius: 4px;
      border: 1px solid #1e8e3e;
      background-color: #ffffff;
      color: #137333;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-toggle:hover { background-color: #f8f9fa; }

    .status-banner.paused .btn-toggle {
      border-color: #1a73e8;
      background-color: #1a73e8;
      color: #ffffff;
    }

    .status-banner.paused .btn-toggle:hover { background-color: #1557b0; }

    .editor-container {
      flex: 1;
      background: #ffffff;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      padding: 24px;
      display: flex;
      flex-direction: column;
    }
    textarea {
      width: 100%;
      height: 100%;
      border: none;
      outline: none;
      resize: none;
      font-size: 32px;
      line-height: 1.8;
      font-family: "Plantagenet Cherokee", "Noto Sans Cherokee", sans-serif;
    }

    footer {
      margin-top: 15px;
      font-size: 13px;
      color: #666;
      display: flex;
      justify-content: space-between;
    }
  </style>
</head>
<body>

  <header>
    <h1>Cherokee Syllabary Transcriber</h1>
    <select id="mic-select">
      <option>Default Microphone</option>
    </select>
  </header>

  <div class="status-banner" id="status-banner">
    <div class="status-left">
      <div class="status-indicator" id="status-dot"></div>
      <span id="status-text">LISTENING — Speak a word syllable-by-syllable...</span>
    </div>
    <button id="toggle-mic-btn" class="btn-toggle" onclick="toggleListening()">
      Pause Listening
    </button>
  </div>

  <div class="editor-container">
    <textarea id="transcript" placeholder="Transcribed text will appear here..." autofocus></textarea>
  </div>

  <footer>
    <span><strong>Spoken Commands:</strong> Say "Delete" to erase last word | Say "Clear All" to reset</span>
    <span>Auto-copies to clipboard on every word</span>
  </footer>

</body>
</html>

```

---

## 6. Detailed Functional Requirements

| Feature ID | Feature Name | Description |
| --- | --- | --- |
| **FR-01** | Silence-Based Segmentation | JS VAD must automatically slice audio upon detecting 600–800ms of silence following speech. |
| **FR-02** | Local Model Inference | Python backend must process incoming audio chunks locally within <300ms latency. |
| **FR-03** | Auto-Appending Canvas | Transcribed Cherokee characters must immediately append to the document canvas. |
| **FR-04** | Hands-Free Control Commands | Basic speech commands (e.g., "Delete", "Clear") must execute document editing actions without requiring mouse interaction. |
| **FR-05** | Listening State Toggle | Users or assistants can manually pause/resume mic listening via UI button. |
| **FR-06** | Auto-Clipboard Sync | Application automatically updates the OS clipboard whenever a new word is committed. |

---

## 7. Packaging & Cross-Platform Distribution Strategy

* **Executable Compiles:** Build target-specific binaries (`.exe` for Windows, executable bundles for macOS and Linux) using **PyInstaller** or **Nuitka**.
* **Direct `venv` Portability Avoidance:** Virtual environments will not be shipped raw due to binary platform dependencies (`.so` / `.pyd` files).
* **OS Entitlements (macOS):** Include `Info.plist` with `NSMicrophoneUsageDescription` during macOS builds to avoid OS-level microphone blocking.

---

## 8. Milestone Roadmap

### Phase 1: Prototype Core Engine

* Set up `pywebview` with local HTTP server context.
* Integrate `@ricky0123/vad-web` in JS and establish two-way communication with Python.
* Mock ASR backend to verify audio buffer transfer and character rendering.

### Phase 2: ASR & Interface Integration

* Load the trained Cherokee Syllable ASR model into the Python backend.
* Implement clean HTML/CSS layout (32px typography, status banner, listening toggle).
* Integrate background clipboard auto-sync.

### Phase 3: Spoken Commands & Refinement

* Add audio keyword triggers for "Delete" (remove last word) and "Clear All".
* Optimize VAD sensitivity (`positiveSpeechThreshold` and `redemptionFrames`) to prevent false triggers.

### Phase 4: Packaging & Testing

* Build macOS, Windows, and Linux standalone executables via PyInstaller.
* Perform user acceptance testing with Cherokee speakers and accessibility testers.
