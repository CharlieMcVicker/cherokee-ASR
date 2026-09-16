---
id: doc-8
title: Syllabary-Guided vs Greedy CTC Segmentation Progress Tracker
type: guide
created_date: '2026-09-16 17:09'
updated_date: '2026-09-16 17:09'
---
# Syllabary-Guided vs Greedy CTC Segmentation Progress Tracker

Tracking comparative accuracy benchmarks and error breakdowns across Cherokee Syllabary dataset splits (\`split_audio_syl_target.csv\`, 1,389 samples) evaluating Greedy ASR inference vs. Syllabary-Guided CTC Segmentation (Relative Contrastive Acoustic Gating, zero tuned hyperparameters).

---

## 1. Latest Evaluation Benchmark (Post-Deaffrication & Laryngeal Reform)

\`\`\`
========================================================================================
        EVALUATION RESULTS SUMMARY: GREEDY VS SYLLABARY-GUIDED CTC SEGMENTATION         
     (ctc-segmentation PR #7 Relative Contrastive Gating - Zero Tuned Hyperparams)      
========================================================================================
Split    | Condition | Greedy CER | Guided CER | Δ CER    | Greedy WER | Guided WER | Δ WER   
----------------------------------------------------------------------------------------
test     | Clean  |      2.02% |      2.32% |   +0.30% |     14.29% |     14.92% |   +0.63%
test     | Noisy  |      4.25% |      2.34% |   -1.91% |     27.10% |     16.18% |  -10.92% 🏆
----------------------------------------------------------------------------------------
valid    | Clean  |      2.16% |      2.28% |   +0.12% |     15.26% |     14.52% |   -0.74% 🏆
valid    | Noisy  |      5.50% |      2.57% |   -2.93% |     32.35% |     18.38% |  -13.97% 🏆
----------------------------------------------------------------------------------------
train    | Clean  |      0.04% |      1.89% |   +1.85% |      0.23% |     12.21% |  +11.98%
train    | Noisy  |      2.11% |      1.77% |   -0.34% |     12.32% |     13.15% |   +0.83% 🏆
----------------------------------------------------------------------------------------
all      | Clean  |      0.42% |      1.97% |   +1.55% |      3.05% |     12.69% |   +9.64%
all      | Noisy  |      2.64% |      1.90% |   -0.74% |     15.72% |     13.96% |   -1.76% 🏆
----------------------------------------------------------------------------------------
========================================================================================
\`\`\`

---

## 2. Iteration History & Milestones

| Stage | Clean Test CER | Clean Valid CER | Clean All CER | Noisy Test CER | Noisy Valid CER | Noisy All CER | Key Structural Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Initial Baseline** | 7.81% | 6.58% | 7.16% | 7.85% | 7.20% | 7.21% | Baseline heuristic penalties. Terminal vowels deleted before pad; spurious \`h\` intrusions everywhere. |
| **1. *HH Constraint & Stride 4** (\`TASK-337\`, \`TASK-340\`) | 3.24% | 3.00% | 2.99% | 3.44% | 3.40% | 3.00% | Blocked intrusive \`h\` before \`hs\`; licensed pre-consonantal \`th\`, \`kh\`, \`lh\`, \`nh\`, \`wh\`, \`yh\`; onset-only syncope. Missing \`h\` dropped 82%. |
| **2. Lateral Deaffrication** (\`TASK-331\`) | **2.32%** | **2.28%** | **1.97%** | **2.34%** | **2.57%** | **1.90%** | Licensed optional stop \`t\` syncope deletion in \`tl\` and \`tlh\` clusters. Lateral errors dropped 89.5% (497 -> 52). Guided decisively beats Greedy on noisy audio. |

---

## 3. Error Breakdown Progression

| Error Taxonomy Category | Baseline | Post-Stride 4 & Masking | Post-Deaffrication (\`tl\` -> \`lh\`) | Current Status |
| :--- | :--- | :--- | :--- | :--- |
| **Terminal Vowel Dropping** | 610 | 12 | 12 | Resolved via onset-only syncope gating. |
| **Spurious Pre-aspiration (\`*hh\`)** | 879 | 153 | 153 | Resolved via \`*HH\` phonotactic masking on \`hs\`. |
| **Lateral Dialect Alternation (\`tl\` vs \`lh\`)** | 497 | 497 | **52** | Reduced by 89.5% via \`tl\` leading \`t\` syncope licensing. |
| **Missing Sonorant Pre-Aspiration (\`nh, wh, yh\`)** | ~140 | ~100 | ~83 | In progress. |
| **Spelling Inconsistencies & Minor Vocalic Syncope** | ~200 | ~120 | ~100 | In progress. |

---

## 4. Evaluation Execution

To reproduce the benchmark:
\`\`\`bash
# Executes across all 1,389 samples using cached LPZ matrices in ~2.7s:
python scripts/rescore_syllabary_dataset.py
\`\`\`
