# Addressed State Attention (ASA): A Slot-Routing Primitive for Sequence Models

## Abstract

Addressed State Attention (ASA) routes token information through a small set of
learned slots that persist across sequence positions. This paper accompanies the
open-source release of ASA/ASM and documents the public implementation, which
emphasizes checkpoint compatibility and reproducibility over maximal performance.

## Introduction

Sequence models typically use attention to mix token representations directly.
ASA introduces a middle layer of slot states that collect information from
writes and serve it back via reads. The slots act as an addressed memory bank
that can compress and route information throughout the sequence.

## Method

### Core mechanism

1. **Write path**: token keys/values attend to a learned set of slot keys.
2. **Slot state**: per-head slot states aggregate token values based on write
   weights.
3. **Read path**: token queries attend over slot states to produce read outputs.
4. **Output projection**: read outputs are projected back into token space.

### Optional components (as implemented)

- **Content read**: an additional read path attends directly over token keys and
  blends with slot reads via a learned gate (`_content_read_gamma_raw`).
- **Slotspace refinement**: slot states can be refined with slot-to-slot
  attention and blended with a learned gate (`_slotspace_gate_raw`).

## What is novel here?

- A slot-based routing primitive that separates writes and reads through an
  explicit addressed slot state.
- A public, checkpoint-stable implementation extracted from research notebooks.
- A minimal, reproducible release with runnable scripts and CPU-friendly tests.

## Related Work

This repository does not claim state-of-the-art results. The release focuses on
making ASA/ASM code and experiments reproducible and reusable. Related work
includes general attention mechanisms, slot-based memory, and routing models.

## Experiments

This release includes:

- A smoke test to validate forward/backward passes.
- A tiny training run on a local text sample.
- A Paris-margin probe that compares logits for `P` vs `L` after prompt prefixes.

These scripts are designed for correctness and reproducibility rather than
benchmark performance.

## Limitations and Open Questions

- The public ASA implementation is simplified relative to notebook variants.
- The probe is a lightweight proxy for more detailed routing analyses.
- No large-scale benchmarks or public checkpoints are provided yet.

## Reproducibility Checklist

- [x] Code released under MIT license.
- [x] Minimal CPU-friendly scripts and tests.
- [x] Determinism notes documented.
- [ ] Reference checkpoints and benchmark runs (planned).

