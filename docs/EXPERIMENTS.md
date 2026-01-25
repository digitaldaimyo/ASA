# Experiments Inventory

This inventory is derived from the archived notebooks in `building_blocks/`.

| File | Purpose | Key modules | Key metrics | Artifacts | Status / productized |
| --- | --- | --- | --- | --- | --- |
| `building_blocks/ASAResearch.ipynb` | Training, probing, visualization, intervention analysis | `AddressedStateAttention` (cleaned + safer), drop-in variants with slot masking, `ASMTrainConfig`, `ASMBlock`, `ASMLanguageModel`, probe utilities | Paris-margin, routing logits/weights trajectories, energy basin score (`-log P(target)`), head alignment cosine similarity | Plots for trajectory bundles, simplex/barycentric animations, energy basins, head alignment tables | Core ASA in `src/asa/asa.py`; probe distilled into `scripts/probe_paris_margin.py` |
| `building_blocks/ASATrainTest.ipynb` | Training-focused experiments, efficient/online ASA, training utilities | `AddressedStateAttention` (baseline + online slotspace scan), `ASMTrainConfig`, training loops (resume support) | Training loss, validation perplexity, throughput, gradient accumulation | Checkpoints (`best.pt`), cached token streams, run logs | Training loop distilled into `scripts/train_mini.py`; smoke path in `scripts/smoke.py` |

## Canonical module definitions referenced

- **ASA “cleaned + safer; checkpoint-stable”**: `AddressedStateAttention` cell labeled “cleaned + safer; checkpoint-stable” in `ASAResearch.ipynb`.
- **ASA “training-focused efficient version”**: `AddressedStateAttention` with online slotspace scan in `ASATrainTest.ipynb`.
- **ASA “refine-geometry logging + intervention”**: trajectory/energy basin cells that collect routing logits and intervene on routing distributions in `ASAResearch.ipynb`.
- **ASM config/block/LM**: `ASMTrainConfig`, `ASMBlock`, `ASMLanguageModel` defined in both notebooks.
