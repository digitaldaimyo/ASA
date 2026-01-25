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

## Mini training run

```bash
python scripts/train_mini.py --device cpu --seed 1337 --outdir runs/train_mini --steps 200
```

Expected outputs:
- `runs/train_mini/train_mini.pt`
- `runs/train_mini/metrics.json`

## Paris-margin probe

```bash
python scripts/probe_paris_margin.py --device cpu --seed 1337 --outdir runs/probes
```

Expected outputs:
- `runs/probes/paris_margin.png`
- `runs/probes/paris_margin.json`

## Determinism notes

- Seeds are set via `--seed` in each script.
- Determinism is best-effort on CPU; GPU kernels may introduce nondeterminism.
