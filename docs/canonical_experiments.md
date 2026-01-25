# Canonical Experiments (Public Core)

This document formalizes the **six canonical experiments** required to support
ASA’s three core claims:

1. ASA routing has longer timescales and lower entropy than residual representations.
2. Factual recall is causally dominated by routing, not residual content.
3. Routing operates on a low-dimensional control manifold.

Each experiment is defined in mechanistic terms and is treated as **required**
for the paper’s causal narrative. All other experiments are supporting,
appendix-only, or archival.

## Canonical Experiment 1 — Routing Trajectory Bundle + Energy Basin

- **Purpose:** Characterize routing trajectories and reveal an energy basin tied
  to factual recall, supporting low-dimensional control structure.
- **Intervention:** None (observational), or optional routing-state override for
  basin slicing.
- **Measurement:** Routing-logit trajectories; energy basin metric
  (e.g., `-log P(target)`) over routing grid.
- **Expected outcome:** Routing trajectories cluster into a coherent basin that
  predicts factual logits.
- **Falsification criterion:** No coherent basin structure; routing trajectories
  fail to correlate with factual logits.
- **Figure(s) produced:** Trajectory bundle with energy basin overlay.
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title ASA / ASM: Trajectory Bundle + Energy Basin (B + D combined)`.

## Canonical Experiment 2 — Paris Basin on a Simplex Sphere (A++)

- **Purpose:** Show that factual recall geometry is organized on a low-dimensional
  routing manifold.
- **Intervention:** None (observational).
- **Measurement:** Routing PCA projection onto a simplex sphere; basin contours.
- **Expected outcome:** A low-dimensional basin structure aligned with the Paris
  direction emerges in routing space.
- **Falsification criterion:** No stable basin in routing PCA space; geometry is
  diffuse and unstructured.
- **Figure(s) produced:** Simplex sphere plot with Paris basin contours.
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title COMPLETE REPLACEMENT: “Paris Basin on a Simplex Sphere (A++)”`.

## Canonical Experiment 3 — Paris-Attractor Emergence Across Depth

- **Purpose:** Determine where in depth the routing control manifold emerges and
  whether entropy decreases with depth.
- **Intervention:** None (observational depth sweep).
- **Measurement:** Layer-wise routing geometry metrics (entropy, PCA structure,
  basin alignment).
- **Expected outcome:** Mid/late layers show sharper routing commitment and
  stronger basin structure than early layers.
- **Falsification criterion:** No depth trend; early layers are as structured as
  late layers.
- **Figure(s) produced:** Layer sweep plots (entropy curves + basin alignment).
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title Layer Sweep: Where does the Paris-attractor geometry emerge?`.

## Canonical Experiment 4 — Routing vs Residual Intervention Parity (FD + PCA)

- **Purpose:** Test whether routing interventions produce stronger causal effects
  on factual logits than residual interventions.
- **Intervention:** Finite-difference perturbations in routing PCA space vs
  residual PCA space (matched magnitude).
- **Measurement:** Change in Paris-margin / factual logits under matched
  perturbations.
- **Expected outcome:** Routing perturbations yield larger, monotonic changes
  than residual perturbations.
- **Falsification criterion:** Residual perturbations match or exceed routing
  effects under the same magnitude.
- **Figure(s) produced:** Intervention parity plot (routing vs residual effects).
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title Experiment: Residual-vs-Routing Intervention Parity Test (FD + PCA, late-window centroid)`.

## Canonical Experiment 5 — Residual Controllability with Routing Frozen

- **Purpose:** Establish that residual perturbations are ineffective once routing
  is fixed, showing routing dominance.
- **Intervention:** Freeze routing to baseline; apply residual perturbations.
- **Measurement:** Change in factual logits under residual-only interventions.
- **Expected outcome:** Residual perturbations produce near-zero effect when
  routing is fixed.
- **Falsification criterion:** Residual perturbations produce strong logit shifts
  even with routing fixed.
- **Figure(s) produced:** Frozen-routing causal comparison plot.
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title Experiment 3: Residual controllability with ROUTING FROZEN (decisive A vs B test)`.

## Canonical Experiment 6 — Routing Timescale & Entropy (Policy Inertia)

- **Purpose:** Quantify routing persistence and entropy decay to support the
  timescale claim.
- **Intervention:** None (observational over time and depth).
- **Measurement:** Routing half-life / inertia curves and entropy over time.
- **Expected outcome:** Routing exhibits long half-life and decreasing entropy
  across depth.
- **Falsification criterion:** Routing entropy does not decrease, or half-life is
  indistinguishable from residual dynamics.
- **Figure(s) produced:** Inertia curves + half-life summary plot.
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title Policy Inertia: curves + uncertainty + half-life summary`.

---

## Required Supporting Example (Public Demo)

### Mini Synthetic Fine-Tune Example

- **Purpose:** Provide a minimal public demonstration of slot-driven fine-tuning
  behavior on synthetic data.
- **Intervention:** Construct synthetic fine-tune batches and run short updates
  under the ASA/ASM training loop.
- **Measurement:** Loss curves on synthetic batches and qualitative outputs from
  `crafted gen` / `multigen` (if used).
- **Expected outcome:** Stable loss decrease without NaNs and predictable
  synthetic outputs.
- **Falsification criterion:** Loss diverges or outputs collapse (NaNs / Inf).
- **Figure(s) produced:** Optional tiny loss curve (appendix/demo).
- **Code location:** `building_blocks/ASAResearch.ipynb` —
  `#@title synthetic fine tune data` (plus optional `crafted gen` / `multigen`).

---

## What This Paper Does Not Claim

- It does **not** claim state-of-the-art performance on language modeling
  benchmarks.
- It does **not** claim that ASA fully replaces standard transformer attention
  in all regimes.
- It does **not** claim that routing manifolds are fully characterized across
  all datasets or scales.
- It does **not** claim that Paris-margin is a comprehensive factuality metric.

## Why These Experiments Are Sufficient

The six canonical experiments jointly establish (1) routing persistence and
entropy reduction over depth (Policy Inertia + Layer Sweep), (2) causal
primacy of routing for factual recall (Residual-vs-Routing Parity + Routing
Frozen test), and (3) low-dimensional control structure (Trajectory Bundle +
Paris Basin). Together they provide mechanistic evidence for ASA’s control
manifold without relying on benchmarks or dataset-specific tuning.
