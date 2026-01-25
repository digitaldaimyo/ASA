# Addressed State Attention (ASA)

Addressed State Attention (ASA) is a research primitive that routes token
information through a small set of learned slots. Addressed State Models (ASM)
stack ASA blocks into a language model. This repository extracts the core code
from the original notebooks and packages it as a runnable, reproducible library
with scripts, tests, and CI.

**Status:** research code (release candidate).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run the three entrypoints (all CPU-friendly by default):

```bash
python scripts/smoke.py --device cpu --seed 1337 --outdir runs/smoke
python scripts/train_mini.py --device cpu --seed 1337 --outdir runs/train_mini --steps 200
python scripts/probe_paris_margin.py --device cpu --seed 1337 --outdir runs/probes
```

## What is ASA?

- A slot-based attention primitive that learns a small set of routing anchors.
- Writes aggregate token information into slot states.
- Reads query slot states back into the token stream.
- Optional content-read path attends directly over tokens.
- Optional slotspace refinement updates slots via slot-to-slot attention.

## Repository layout

- `src/asa/`: core ASA/ASM library.
- `scripts/`: runnable entrypoints (smoke, train-mini, probe).
- `tests/`: minimal tests for shapes, checkpoint keys, and NaN checks.
- `docs/`: method notes, experiment inventory, roadmap, assumptions.
- `building_blocks/`: raw notebook archive (immutable).
- `paper/`: draft paper-style exposition of the method.

For full reproducibility guidance, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Release Checklist

- [x] Packaged `asa` library with ASA + ASM implementations.
- [x] Scripts for smoke test, mini training, and Paris-margin probe.
- [x] Minimal tests and CI workflow.
- [x] Documentation for method, experiments, roadmap, assumptions.
- [x] Reproducibility notes and placeholder citation metadata.
- [ ] Train and publish a reference checkpoint for benchmark comparisons.
