---
id: doc-5
title: Noisy ASR Evaluation and Alignment Confusion Matrix Guide
type: guide
created_date: '2026-09-02 19:02'
updated_date: '2026-09-02 19:03'
---
# Noisy ASR Evaluation & Confusion-Driven Alignment Engine Guide

## 1. Executive Summary & Overview

This guide documents the architecture, mathematical formulations, empirical results, and operational usage of the **Noisy ASR Evaluation and Empirical Confusion Cost Engine** for Cherokee ASR and forced alignment.

In pristine elicited conditions, the Cherokee Wav2Vec2 model achieves **0.21% – 0.88% Character Error Rate (CER)**. While excellent for recognition, standard greedy argmax confusion matrices on clean audio are virtually identity matrices: they fail to capture acoustic boundary ambiguity, fast-speech assimilation, or phonological similarity. 

To create a mathematically principled, asymmetric substitution cost function for DP/DTW forced alignment (e.g. `NeedlemanWunschWordAligner`), this module introduces:
1. **Calibrated Multi-Regime Audio Perturbations**: Simulates realistic acoustic degradation across sweeps of +25 dB down to -5 dB SNR.
2. **Top-K Soft-Probability CTC Emission Accumulation**: Collects frame-level posterior distributions rather than single argmax tokens.
3. **Clamped Logarithmic Cost Engine**: Normalizes conditional probabilities into a metric bounded in [0.0, 1.0].
4. **Dedicated 10% Operational Alignment Matrix**: Specifically calibrated on 6,844 utterances to match real-world ~10% CER alignment conditions.

---

## 2. Mathematical & Algorithmic Foundations

### 2.1. RMS-Calibrated Acoustic Degradation
For a 16 kHz mono waveform $x(t)$, degraded audio $\tilde{x}(t)$ is generated with calibrated Signal-to-Noise Ratio (SNR):

$$\alpha = \sqrt{\frac{P_x}{P_\eta \cdot 10^{\text{SNR}_{\text{dB}} / 10}}}$$

$$\tilde{x}(t) = \text{clamp}(x(t) + \alpha \cdot \eta(t), -1.0, 1.0)$$

* **White Noise**: $\eta(t) \sim \mathcal{N}(0, 1)$.
* **Pink Noise**: $1/\sqrt{f}$ spectral roll-off synthesized in frequency domain via `rfft`, magnitude scaling by $1/\sqrt{k+1}$, and `irfft` inversion.
* **Pitch-Preserving Cadence Warp**: Phase vocoder STFT time-stretch ($s \in [0.85, 1.15]$) preserving Cherokee pitch/tonal distinctions without frequency warping.
* **Acoustic Filter**: 4th-order Butterworth bandpass filter (300 Hz – 3400 Hz) simulating low-grade dynamic/laptop mic capture.

### 2.2. Soft-Probability Confusion Accumulation & Dirichlet Prior
Given frame-level CTC softmax posterior distributions $P(c \mid t)$, non-blank emission peaks are aligned with ground-truth reference characters via Wagner-Fischer backtrace. For each hypothesis character, top-K posterior probabilities are accumulated:

$$C[\text{ref}, \text{alt}_k] \leftarrow C[\text{ref}, \text{alt}_k] + P(\text{alt}_k \mid t)$$

Dirichlet smoothing prior (default $\alpha = 0.1$) prevents zero-probability penalties:

$$P(\text{hyp} = j \mid \text{ref} = i) = \frac{C[i, j] + \alpha}{\sum_{k=1}^{|V|} C[i, k] + \alpha \cdot |V|}$$

### 2.3. Clamped Normalized Logarithmic Substitution Cost
To convert conditional confusion probabilities into substitution costs for sequence alignment:

$$d(i, i) = 0.0$$

$$d(i, j) = \text{clip}\left(\frac{\ln(P(j \mid i) + \epsilon)}{\ln \epsilon}, 0.0, 1.0\right) \quad \text{for } i \ne j$$

If a reference token has total empirical observations $N_i < N_{\min}$ (default $N_{\min} = 5$), the cost safely falls back to $1.0$.

---

## 3. Dataset Preprocessing & Cleaning Rules

To prevent data corruption and artificial CER inflation, two critical normalization rules are strictly enforced:

1. **Punctuation & Colon Stripping**:
   In `training_data/processed/cim-wav2vec2-train.csv`, colons (`:`) denote vowel length (e.g. `nikhv: u:we:hsta:neli:to:ho:`). The model was trained with colons stripped via `normalize_text()`. If colons are not removed prior to evaluation, they are all counted as deletions, falsely inflating baseline CER from **0.21% up to 15.1%**!
   `scripts/run_noisy_eval.py` applies the training loop's canonical regex:
   ```python
   chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]"
   apostrophe_variants = r"[’‘ʼʻ`´‛]"
   ```
2. **Bad Data Filtering (`[dg]`)**:
   Sentences containing `'d'` or `'g'` are filtered out on ingestion. The CIM elicitation orthography uses `t` and `k` for stops.
3. **Clean 17-Token Vocabulary**:
   Active alphabet is restricted strictly to:
   `[' ', "'", 'a', 'e', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 's', 't', 'u', 'v', 'w', 'y']`
   Out-of-vocabulary model tokens (`d`, `g`, `~`, `q`) and blank empty strings (`""`) are filtered out of candidate accumulation.

---

## 4. Empirical Evaluation Across Noise Regimes

Evaluation across 37,490 audio runs on `cim-wav2vec2-train.csv`:

| SNR (dB) | Noise Profile | Samples | Mean CER |
| :---: | :---: | :---: | :---: |
| **+25.0 dB** | White | 3,749 | **0.88%** |
| **+25.0 dB** | Pink | 3,749 | **0.21%** |
| **+15.0 dB** | White | 3,749 | **7.72%** |
| **+15.0 dB** | Pink | 3,749 | **2.85%** |
| **+5.0 dB** | White | 3,749 | **33.96%** |
| **+5.0 dB** | Pink | 3,749 | **24.86%** |
| **0.0 dB** | White | 3,749 | **56.56%** |
| **0.0 dB** | Pink | 3,749 | **51.98%** |
| **-5.0 dB** | White | 3,749 | **82.32%** |
| **-5.0 dB** | Pink | 3,749 | **79.94%** |

### Regime Comparison

| Regime | SNRs | Utterances | CER | Graph Components ($\tau = 0.05$) | Phonetic Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Noise** | 25, 15 dB | 14,996 | **2.91%** | **17** | Sharp diagonal; near-zero off-diagonal cross-talk. |
| **Mid Noise** | 5 dB | 7,498 | **29.41%** | **14** | Phonological assimilation: nasals $\{m, n\}$, plosives $\{t, k\}$, high vowels $\{i, u\}$. |
| **High Noise** | 0, -5 dB | 14,996 | **67.70%** | **7** | Severe acoustic masking causes phonetic collapse into 7 broad clusters. |

---

## 5. Dedicated 10% Operational Alignment Matrix

For real-world text alignment where ASR performance operates around **~10% CER**, we extracted a dedicated calibration slice of **6,844 utterances** with an empirical mean CER of **9.93%**:

### Key Substitution Costs at 10% Operational CER:
* **Identical Characters**: `0.0000`
* **High Vowel Confusion ($u \to i$)**: `0.1727`
* **Plosive Confusion ($t \to k$)**: `0.2643`
* **Aspiration / Fricative Confusion ($s \to h$)**: `0.2751`
* **Nasal Confusion ($m \to n$)**: `0.3202`
* **Consonant $\to$ Vowel Distortion ($t \to a$)**: `0.5659`
* **Distant Substitution ($k \to o$)**: `0.8235`

### Information Quality Benchmark on 10% CER Holdout Data:
* **Low Noise Matrix**: Error Perplexity = **249.08** (too conservative, assigns ~1.0 to almost all substitutions).
* **High Noise Matrix**: Error Perplexity = **21.68**, but Phonetic Contrast $\approx 0.00$ ($t \to k$ costs same as $t \to a$).
* **10% Dedicated Matrix**: Optimal balance with Perplexity = **58.20** and strong contrast ($+0.269$ between true confusions and distant pairs).

---

## 6. Artifact Inventory & Formats

All evaluation outputs are persisted under `runs/evaluation/`:

| Artifact File | Format | Description |
| :--- | :--- | :--- |
| `eval_records.jsonl` | JSONL (66 MB) | 37,490 streaming evaluation records with top-K probabilities and CER. Serves as on-disk resume cache. |
| `confusion_cost_matrix_10pct.json` | JSON | **Primary production cost metric** for forced alignment at ~10% operational CER. |
| `confusion_matrix_10pct.csv` | CSV | 17x17 conditional probability matrix for 10% operational regime. |
| `confusion_vs_cost_heatmap_10pct.png` | PNG | Side-by-side 2D heatmap plot for 10% operational regime. |
| `confusion_cost_matrix_low.json` | JSON | Cost matrix for Low Noise regime (2.91% CER). |
| `confusion_cost_matrix_mid.json` | JSON | Cost matrix for Mid Noise regime (29.41% CER). |
| `confusion_cost_matrix_high.json` | JSON | Cost matrix for High Noise regime (67.70% CER). |
| `confusion_vs_cost_heatmaps_by_regime.png` | PNG | 3-row comparative multi-panel grid comparing Low, Mid, and High regimes. |
| `snr_drift.png` | PNG | Phonetic cluster stability and CER curve as a function of SNR. |
| `confusion_manifold_3d.png` | PNG | 3D surface plot over the 17-character vocabulary. |

---

## 7. CLI Usage & Reproducibility

All tools leverage on-disk caching and run without redundant neural network forward passes:

```bash
# 1. Run or resume full evaluation sweep across all 10 tiers (~30 min first time, instant on resume):
python scripts/run_noisy_eval.py --dataset-csv training_data/processed/cim-wav2vec2-train.csv

# 2. Re-partition Low, Mid, and High regimes (~14 seconds):
python scripts/generate_regime_heatmaps.py

# 3. Generate dedicated 10% operational alignment matrix (~7 seconds):
python scripts/create_10pct_alignment_matrix.py
```

### Loading in Python Aligner Code:
```python
from transcription.alignment import ConfusionMatrixCostMetric, NeedlemanWunschWordAligner

# Load dedicated 10% operational metric
metric = ConfusionMatrixCostMetric.from_json("runs/evaluation/confusion_cost_matrix_10pct.json")

# Configure aligner
aligner = NeedlemanWunschWordAligner(
    distance_metric=metric,
    gap_cost=0.8,
    fuse_penalty=0.15,
)
```
