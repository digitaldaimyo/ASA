# Reproducibility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

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
