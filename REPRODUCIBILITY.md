# Reproducibility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Reference checkpoint download

```python
from huggingface_hub import hf_hub_download

repo_id = "DigitalShogun/ASA-ASM-wikitext103-raw"
filename = "ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt"
path = hf_hub_download(repo_id=repo_id, filename=filename)
```

Notes:
- CPU-only inference is supported for the provided demos and probes.
- Set `HF_TOKEN` if the repository is private.

## Smoke test

```bash
python scripts/smoke.py --device cpu --seed 1337 --outdir runs/smoke
```

Expected outputs:
- `runs/smoke/smoke.pt`
- Console line with the final loss.

Expected runtime (CPU): < 30s.

## Mini training run

```bash
python scripts/train_mini.py --device cpu --seed 1337 --outdir runs/train_mini --steps 200
```

Expected outputs:
- `runs/train_mini/train_mini.pt`
- `runs/train_mini/metrics.json`

Expected runtime (CPU): ~1-2 minutes.

## Paris-margin probe

```bash
python scripts/probe_paris_margin.py --device cpu --seed 1337 --outdir runs/probes
```

Expected outputs:
- `runs/probes/paris_margin.png`
- `runs/probes/paris_margin.json`

Expected runtime (CPU): < 30s.

## Determinism notes

- Seeds are set via `--seed` in each script.
- Determinism is best-effort on CPU; GPU kernels may introduce nondeterminism.

## Hardware assumptions

- CPU-only runs are supported (no GPU required).
- Memory footprint is small (<1 GB) for the provided configs.

## Probe output notes

The Paris-margin probe uses a simple character-level tokenizer. The margin is
computed at the final prompt position between the logits for `P` and `L`, which
serves as a lightweight stand-in for the notebook’s Paris-vs-London analysis.
