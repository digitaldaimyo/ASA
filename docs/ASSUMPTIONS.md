# Assumptions

- The public release uses a simplified ASA implementation that preserves
  checkpoint-critical parameter names but does not replicate every notebook
  optimization.
- Default training scripts rely on a small local text sample to avoid mandatory
  external downloads.
- Probe scripts operate on randomly initialized or locally trained models unless
  a checkpoint path is provided.
- Output directories default to `runs/` under the repo root.
