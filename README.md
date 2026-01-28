# Addressed State Attention (ASA)

Addressed State Attention (ASA) is a research primitive that routes token
information through a small set of learned slots. Addressed State Models (ASM)
stack ASA blocks into a language model. This repository extracts the core code
from the original notebooks and packages it as a runnable, reproducible library
with scripts, tests, and CI.

**Status:** research code (release candidate).

## Colab Notebooks

- **Functional Validation Quickstart**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitaldaimyo/ASA/blob/main/notebooks/ASA_Colab_Quickstart.ipynb)

- **HF Checkpoint + Canon Probes + Mini Finetune**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitaldaimyo/ASA/blob/main/notebooks/ASA_HF_Checkpoint_CanonProbes_and_MiniFinetune.ipynb)

- **Mechanistic Probing (Reorganized Narrative)**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitaldaimyo/ASA/blob/main/notebooks/ASA_Mechanistic_Probing_Reorg.ipynb)

What this notebook validates:
- Core forward-pass shape invariants across ASA variants.
- Gradient flow and parameter updates on a tiny optimization loop.
- Seeded determinism on CPU (eval mode).
- Slot/head sweep to ensure no shape crashes.
- Slot masking behavior and finite outputs.
- Routing override hook behavior (if exposed).
- Online and intervention variant toggles.

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

## Reference checkpoint

The v0.1.x release artifacts reference a public ASM checkpoint trained on
WikiText-103 (raw) using the baseline ASA/ASM configuration.

- **HF repo:** `DigitalShogun/ASA-ASM-wikitext103-raw`
- **Checkpoint filename:** `ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt`
- **What it enables:** running canonical probes and the Colab notebooks
- **Compatibility:** the loader normalizes legacy naming; CPU inference is
  supported for small demos

## Hugging Face Demo (CPU)

Run a smoke inference pass using the published checkpoint:

```bash
python -m asa.demo_hf --repo DigitalShogun/ASA-ASM-wikitext103-raw
```

Or load it in Python with checkpoint compatibility helpers:

```python
from asa.load_pretrained import load_pretrained

model, report, cfg = load_pretrained(
    repo_id="DigitalShogun/ASA-ASM-wikitext103-raw",
    filename=None,
    variant="baseline",
    device="cpu",
)
```

Expected output (abbreviated):

```
Downloading from DigitalShogun/ASA-ASM-wikitext103-raw
Checkpoint state dict source: model
Loaded checkpoint with strict=True
PASS: forward logits shape + finite
Generated 20 tokens. Final shape: (1, 52)
```

Troubleshooting:
- If the HF repo is private, set `HF_TOKEN` in your environment.
- If downloads are slow, check the Hugging Face cache at `~/.cache/huggingface`.
- Use `--ckpt` to point to a specific filename in the repo.

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
  - Upload `paper/latex/` to Overleaf and set `main.tex` as the root file.

For full reproducibility guidance, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Release Checklist

- [x] Packaged `asa` library with ASA + ASM implementations.
- [x] Scripts for smoke test, mini training, and Paris-margin probe.
- [x] Minimal tests and CI workflow.
- [x] Documentation for method, experiments, roadmap, assumptions.
- [x] Reproducibility notes and citation metadata.
- [x] Publish a reference checkpoint for canonical probes.
