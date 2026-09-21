# Technical Specification: Per-Token Intrusion Penalties and Explicit Syncope/Intrusion Masks for `ctc-segmentation`

**Version:** 3.0  
**Target Package:** `ctc-segmentation` (Cython core: `ctc_segmentation_dyn.pyx`, Python API: `ctc_segmentation.py`)  
**Status:** Ready for Implementation & Handoff  

---

## 1. Executive Summary & Problem Statement

In CTC-based forced alignment with phonological variation (such as Cherokee New Testament alignment), the aligner must support two distinct types of structural deviations from canonical citation text:
1. **Syncope (Omission):** A vowel present in the ground-truth transcript is elided in fast or natural speech.
2. **Intrusive Detours (Insertion):** Unwritten phonetic segments (such as pre-aspiration `/h/`, laryngeal sonorants `/nh, lh, wh, yh/`, or glottal stops `/'/`) occur acoustically but are absent from the ground-truth transcript.

### 1.1 The Acoustic Asymmetry Problem with Uniform Intrusion Penalties
Currently, `CtcSegmentationParameters` accepts a single global float `intrusive_penalty: float`.
* **/h/ (Aspiration / Breathiness):** Has broad spectral energy spanning 30–100ms+ and frequently registers modest baseline acoustic log-probabilities. It requires a **higher penalty threshold** (e.g., `3.5 – 5.0`) to avoid triggering on trailing breath, consonant release noise, or background frication.
* **/'/ (Glottal Stop):** A brief, transient closure event (10–20ms / 1–2 frames) characterized by an amplitude drop. Because the acoustic window is brief, a uniform penalty of `2.5` almost completely silences real glottal stops. Setting a lower penalty (e.g., `0.5`) to catch glottal stops causes `/h/` to flood every syllable.

### 1.2 The Semantic Separation Problem: Syncope vs. Intrusive Transitions
Skipping an unwritten intrusive candidate must **not** be treated as a case of syncope:
* **Syncope** represents the omission of an *actual ground-truth token* from the canonical spelling and incurs `syncope_penalty`.
* **Intrusion** represents a *detour insertion* into an acoustic frame not present in the canonical spelling and incurs `intrusive_penalty[token]`.
* An alignment step that does *not* insert an intrusive detour simply proceeds along the canonical ground-truth path with **zero penalty**.

To enable language-specific phonotactic guidance without contaminating the core C++/Cython engine with language-specific rules, `ctc-segmentation` must accept:
1. **Per-token intrusion costs** (`intrusive_penalties: Dict[str, float]` or 1D float array).
2. **Explicit `syncope_mask`** (marking ground-truth columns where vowel deletion is permitted).
3. **Explicit `intrusion_mask` / `intrusion_site_mask`** (marking ground-truth transition sites where intrusive detours are phonotactically licensed).

---

## 2. Configuration & Parameter Architecture

### 2.1 Updates to `CtcSegmentationParameters` (`ctc_segmentation.py`)

```python
class CtcSegmentationParameters:
    # --- Existing Syncope Parameters ---
    syncope_tokens: Optional[Sequence[Union[str, int]]] = None
    syncope_penalty: float = 2.0
    is_syncope_token: Optional[np.ndarray] = None  # 1D int8 array of size len(ground_truth)

    # --- New / Updated Intrusive Parameters ---
    intrusive_tokens: Optional[Sequence[Union[str, int]]] = None
    
    # Can be a single float (backwards compatible) or a per-token mapping:
    # e.g., {"h": 4.5, "'": 0.8}
    intrusive_penalties: Optional[Union[float, Dict[Union[str, int], float]]] = None
    
    # Internal 1D float32 array aligned with intrusive_token_ids
    intrusive_penalty_array: Optional[np.ndarray] = None

    # Optional 1D int8 array of size len(ground_truth) gating transition sites:
    # is_intrusive_site[c] == 1 allows intrusive detour transitions between c-1 and c
    # is_intrusive_site[c] == 0 forbids intrusive detour transitions at column c
    is_intrusive_site: Optional[np.ndarray] = None
```

### 2.2 Parameter Semantics & Parsing
* If `intrusive_penalties` is provided as a `float`, all intrusive tokens inherit that uniform penalty (100% backwards compatible).
* If `intrusive_penalties` is provided as a `dict` (e.g. `{"h": 4.5, "'": 0.8}`), `_parse_intrusive_penalties()` constructs a 1D `np.float32` array `intrusive_penalty_array` where index `k` holds the penalty for `intrusive_token_ids[k]`.
* If `is_intrusive_site` is `None`, all column transitions $c \ge 1$ are eligible for intrusive detours (default behavior).
* If `is_intrusive_site` is provided, only columns where `is_intrusive_site[c] == 1` will evaluate intrusive transitions.

---

## 3. Mathematical Formulation (Forward DP Trellis)

Let:
* $t \in [0, T-1]$ be the audio frame time index.
* $c \in [0, L-1]$ be the ground-truth sequence column index.
* $\text{lpz}[t, v]$ be the log-probability of token $v$ at frame $t$.
* $J_{\text{intrusive}} = [j_0, j_1, \dots, j_{K-1}]$ be the array of intrusive token IDs.
* $\boldsymbol{\lambda}_{\text{intrusive}} = [\lambda_0, \lambda_1, \dots, \lambda_{K-1}]$ be the per-token intrusive penalties.
* $\mathbf{M}_{\text{site}}[c] \in \{0, 1\}$ be the intrusive site mask.
* $\mathbf{M}_{\text{syncope}}[c] \in \{0, 1\}$ be the syncope token mask.

### 3.1 Syncope Transition (Unchanged Semantics)
Syncope transitions are evaluated only when $\mathbf{M}_{\text{syncope}}[c-1] == 1$ (or Case B with trailing blank):
$$\text{prob}_{\text{syncope}}(t, c) = \max_{s} \left( \text{table}[t-1 + \Delta_{\text{offset}}, c - 2] + \text{lpz}[t + \text{offset}, \text{ground\_truth}[c, s]] \right) - \lambda_{\text{syncope}}$$

### 3.2 Per-Token Intrusive Transition with Site Masking
Intrusive detour transitions are evaluated only when:
$$K > 0 \quad \text{and} \quad c \ge 1 \quad \text{and} \quad t \ge 2 \quad \text{and} \quad \mathbf{M}_{\text{site}}[c] == 1$$

For each candidate predecessor distance $s \ge 0$, blank stride $\delta \in [0, \delta_{\text{max}}]$, and intrusive token $k \in [0, K-1]$:

$$\text{base\_prob} = \text{table}[t - 2 - \delta + \Delta_{\text{offset}}, c - 1 - s] + \sum_{b=t-\delta}^{t-1} \text{lpz}[b + \text{offset}, \text{blank}] + \text{lpz}[t + \text{offset}, \text{ground\_truth}[c, s]]$$

$$\text{cand\_prob}(k) = \text{base\_prob} + \text{lpz}[t - 1 - \delta + \text{offset}, j_k] - \boldsymbol{\lambda}_{\text{intrusive}}[k]$$

$$\text{prob}_{\text{intrusive}}(t, c) = \max_{s, \delta, k} \text{cand\_prob}(k)$$

---

## 4. Cython Core Implementation (`ctc_segmentation_dyn.pyx`)

```cython
def cython_fill_table(np.ndarray[np.float32_t, ndim=2] table,
                      np.ndarray[np.float32_t, ndim=2] lpz,
                      np.ndarray[np.int64_t, ndim=2] ground_truth,
                      np.ndarray[np.int64_t, ndim=1] offsets,
                      np.ndarray[np.int8_t, ndim=1] is_syncope_token,
                      float syncope_penalty,
                      np.ndarray[np.int64_t, ndim=1] intrusive_token_ids,
                      np.ndarray[np.float32_t, ndim=1] intrusive_penalties,
                      np.ndarray[np.int8_t, ndim=1] is_intrusive_site,
                      int intrusive_max_stride,
                      int blank,
                      int flags):
    # ... setup and window offsets ...

    # Inside column loop (c) and time loop (t):
    # Evaluate intrusive detour transitions:
    intrusive_prob = prob_max
    if num_intrusive_tokens > 0 and c >= 1 and t >= 2:
        if is_intrusive_site.shape[0] == 0 or is_intrusive_site[c] == 1:
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] == -1 or c - 1 - s < 0:
                    continue
                delta_offset = offset_sum - offsets[c - 1 - s]
                for delta in range(0, min(intrusive_max_stride + 1, t - 1)):
                    t_prev = t - 2 - delta + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        blank_sum = 0.0
                        for b_t in range(t - delta, t):
                            blank_sum += lpz[b_t + offset_sum, blank]
                        base_prob = (
                            table[t_prev, c - 1 - s]
                            + blank_sum
                            + lpz[t + offset_sum, ground_truth[c, s]]
                        )
                        for j_idx in range(num_intrusive_tokens):
                            j = intrusive_token_ids[j_idx]
                            p_cand = base_prob + lpz[t - 1 - delta + offset_sum, j] - intrusive_penalties[j_idx]
                            if p_cand > intrusive_prob:
                                intrusive_prob = p_cand
```

---

## 5. Traceback & Backtracking Updates (`ctc_segmentation.py`)

In the backtracking traceback loop:
```python
for j_idx in range(num_intrusive_tokens):
    j = intrusive_token_ids[j_idx]
    detour_score = (
        table[t_prev, c - 1 - s]
        + lpz[t - 1 - delta + offsets[c], j]
        - intrusive_penalties[j_idx]
        + blank_sum
        + lpz[offsets[c] + t, ground_truth[c, s]]
    )
    if np.isclose(table[t, c], detour_score, atol=1e-3):
        # Emit intrusive character at frame (t - 1 - delta)
        state_list[offsets[c] + t - 1 - delta] = config.char_list[j]
        char_probs[offsets[c] + t - 1 - delta] = lpz[offsets[c] + t - 1 - delta, j]
        # Step back trellis state pointers
        c -= 1 + s
        t -= 2 + delta - (offsets[c + 1 + s] - offsets[c])
        found_transition = True
        break
```

---

## 6. Verification & Test Plan

1. **Unit Test: Per-Token Penalty Sensitivity**
   - Provide synthetic audio with acoustic evidence for both `/h/` and `/'/`.
   - Set `intrusive_penalties={"h": 10.0, "'": 0.1}`: assert `/'/` is inserted and `/h/` is suppressed.
   - Set `intrusive_penalties={"h": 0.1, "'": 10.0}`: assert `/h/` is inserted and `/'/` is suppressed.
2. **Unit Test: Site Mask Constraint (`is_intrusive_site`)**
   - Provide high acoustic likelihood for `/'/` across all frames.
   - Set `is_intrusive_site[3] = 1` and `is_intrusive_site[k] = 0` for $k \ne 3$.
   - Assert `/'/` is emitted exclusively at column 3 and nowhere else.
3. **Unit Test: Backwards Compatibility**
   - Setting `intrusive_penalty=0.5` as a float produces identical outputs and timings to baseline.
