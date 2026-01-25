# Addressed State Attention (ASA): A Routing-Control Primitive for Sequence Models

## STEERING NOTE TO CODEX

Treat this document as the authoritative conceptual reference for the ASA project.
When cleaning notebooks, structuring the repo, or preparing release artifacts,
prefer interpretations and terminology from this document over older comments or
filenames. The goal is clarity of mechanism and causal role, not benchmark
chasing.

## Abstract

Addressed State Attention (ASA) is an attention primitive that introduces a
persistent, low-dimensional routing state into sequence models. Tokens do not
attend to each other directly; instead, they write into and read from a small
set of addressed slot states that persist across sequence positions and layers.
We release a public implementation of ASA and its language-model variant (ASM),
along with reproducible analysis tools. Using controlled finite-difference
interventions, we show that factual recall is causally dominated by routing
dynamics rather than residual token representations. Perturbing routing states
reliably steers factual logits, while equivalent perturbations to residual
activations have negligible effect once routing is fixed. These results suggest
that factual recall in ASA-based models emerges from trajectories through a
routing control manifold, rather than from static content storage. ASA thus
provides both a modeling primitive and an analysis lens for studying memory,
inertia, and controllability in sequence models.

## 1. Introduction

Attention mechanisms typically entangle where information flows with what
information is stored. Tokens attend directly to other tokens, and the same
operation both aggregates content and determines routing. While effective, this
entanglement obscures the internal dynamics of memory persistence, factual
recall, and control over generation.

Addressed State Attention (ASA) separates these concerns by introducing an
explicit routing substrate: a small set of slot states that persist across
sequence positions. Tokens write information into slots and later read from
them. Routing decisions are mediated by slot addressing rather than direct
token-to-token interaction.

This separation enables two outcomes:

- **New modeling behavior:** information can persist and accumulate across time
  in a structured, low-dimensional state.
- **New analysis capability:** routing can be studied, intervened on, and
  causally compared to residual representations.

The work presented here focuses on the second outcome. We show that routing
states exhibit longer timescales, lower intrinsic dimensionality, and stronger
causal influence over factual outputs than token-level representations.

## 2. Addressed State Attention

### 2.1 Core mechanism

At each layer, ASA maintains a set of per-head slot states. A forward pass
consists of four conceptual stages:

1. **Write path**
   Token keys attend over learned slot keys, producing write weights.
2. **Slot state update**
   Slot states aggregate token values using these write weights. Slot states
   persist across sequence positions, allowing information to accumulate.
3. **Read path**
   Token queries attend over slot states to produce read vectors.
4. **Output projection**
   Read vectors are projected back into token space and combined with the
   residual stream.

Crucially, tokens never attend to other tokens directly. All information flow
is mediated through the slot states.

### 2.2 Optional components in the public implementation

The released implementation exposes several optional components that were used
in exploratory research and are included for completeness and reproducibility:

- **Content read path**
  A parallel token-to-token attention path blended with slot reads via a learned
  gate. This allows partial fallback to conventional attention behavior.
- **Slotspace refinement**
  Slot-to-slot attention that allows slot states to exchange information,
  controlled by a learned scalar gate.

### STEERING NOTE TO CODEX

When organizing code and docs, treat these components as extensions, not the
conceptual core. The core ASA primitive is the write → slot → read loop.

## 3. ASA as a Control System

While ASA can be described as a memory mechanism, our experiments support a
stronger interpretation: slot routing functions as a control system over
content flow.

Across layers and sequence positions, routing states exhibit:

- **Low intrinsic dimensionality**
  Late-window routing trajectories are often dominated by 1–3 principal
  components.
- **Longer timescales**
  Routing states evolve more slowly than token representations and persist
  across positions.
- **Entropy reduction across depth**
  Routing distributions become more committed in mid-to-late layers.
- **Coherent control fields**
  Gradients of factual logits align consistently with routing directions.

In contrast, residual hidden states exhibit higher dimensionality and weaker
causal influence once routing is fixed. These observations are characteristic
of a control manifold rather than a static memory lookup.

## 4. Experiments

### STEERING NOTE TO CODEX

The experiments in the notebooks are exploratory and heterogeneous. When
refactoring, extract representative, minimal versions of each experiment below
rather than attempting to preserve every variant.

### 4.1 Sanity and reproducibility tests

We provide CPU-friendly tests that validate:

- Forward and backward passes
- Shape invariants
- Non-degenerate gradients
- Determinism under fixed seeds
- Masking behavior
- Routing override hooks
- Intervention toggles

These tests serve as executable documentation of the ASA control surface and
are intended to run in constrained environments (e.g., Colab CPU).

### 4.2 Factual probe: Paris-margin

To study factual recall, we use a simple contrastive metric evaluated after
prompts referencing France. This probe is not intended as a benchmark. Instead,
it provides a scalar signal that is sensitive to factual correctness and
amenable to controlled intervention.

### 4.3 Routing geometry across layers

We analyze late-window routing states using PCA and clustering:

- Early layers exhibit diffuse, rotating routing mass.
- Mid layers show entropy reduction and partial alignment.
- Late layers exhibit coherent, low-dimensional structure despite reduced
  explained variance.

This indicates increasing commitment and specialization of routing behavior
across depth.

### 4.4 Finite-difference Jacobian analysis

We estimate gradients of the Paris-margin with respect to:

- Routing logits (projected into routing PCA space)
- Residual hidden states (projected into residual PCA space)

Key findings:

- Early layers: gradients align strongly with the dominant routing PC.
- Mid layers: gradients rotate across routing dimensions.
- Late layers: gradients spread but remain coherent within routing space.
- Residual gradients show substantially weaker and less consistent effects.

### 4.5 Matched-effect interventions: routing vs residual

To directly compare causal efficiency, we perform matched interventions:

- **Routing intervention:** override routing weights along an estimated control
  direction.
- **Residual intervention:** inject perturbations into hidden states while
  freezing routing to baseline.

Results:

- Routing perturbations produce monotonic, linear changes in factual logits.
- Residual perturbations produce near-zero effect under frozen routing.
- The disparity grows in later layers.

This establishes routing as the dominant causal pathway for factual recall in
ASA-based models.

## 5. Implications

These results support a view in which:

- Facts are not stored as static vectors.
- Recall emerges from trajectories through routing space.
- Late-layer control dominates early content encoding.

This perspective has implications for:

- **Interpretability:** routing states provide a compact control representation.
- **Fine-tuning:** modifying routing may be more effective than modifying
  content.
- **Hallucination analysis:** failures may reflect control misalignment rather
  than missing information.
- **Model design:** separating control from content may improve robustness and
  controllability.

## 6. Limitations and Open Questions

- No large-scale benchmarks are reported.
- The Paris-margin probe is intentionally simple.
- The relationship between ASA routing and standard transformer attention
  remains to be formally characterized.
- Stability and scaling behavior at very long context lengths remain open.

## 7. Reproducibility and Release

This release includes:

- MIT-licensed source code
- CPU-friendly sanity tests and demos
- A public pretrained checkpoint
- Analysis notebooks documenting routing dynamics

### STEERING NOTE TO CODEX

The repo should prioritize:

- Clear separation between library code, experiments, and analysis
- Runnable demos over archival completeness
- Truth-first naming (architecture parameters over historical tags)

## 8. Conclusion

Addressed State Attention introduces a persistent routing substrate that
separates control from content in sequence models. Through controlled
interventions, we show that factual recall in ASA-based models is primarily
mediated by routing dynamics rather than residual representations. ASA thus
provides both a modeling primitive and a framework for studying how sequence
models store, retrieve, and control information over time.

## Appendix A: Guidance for Repository Organization

### STEERING NOTE TO CODEX

Suggested structure:

```
asa/
  core.py           # minimal ASA primitive
  variants.py       # online / intervene variants
  utils.py

experiments/
  sanity/
  paris_margin/
  routing_vs_residual/

notebooks/
  exploratory/      # raw research notebooks (archived)
  cleaned/          # distilled analysis notebooks

docs/
  paper.md
  figures/

tests/
  test_basic.py
  test_masking.py
```

## Appendix B: Author Context (Optional for arXiv)

This work was conducted independently using publicly available tools and
compute. Findings were vetted via reproducible experiments and cross-checked
using multiple language models as stand-ins for traditional peer discussion.
