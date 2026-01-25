CODEx BRIEF — ASA/ASM repo extraction + release prep (single long session)
Mission: Convert a messy research notebook dump into a clean, public, reproducible open-source repo for Addressed State Attention (ASA) and Addressed State Models (ASM). You must inspect notebooks to learn what exists, document the experiment span, and extract core code into a cohesive package while preserving original notebooks unchanged in an archival folder.
Non-negotiables
Do not edit files inside /building_blocks/ except adding a short README.md within that folder explaining it is an archive. Treat it as immutable source evidence.
Create a clean public-facing library + scripts. Notebooks become optional demos, not the primary interface.
Preserve checkpoint compatibility: do not rename checkpoint-critical parameter/buffer names shown in notebooks (e.g., slot_keys, Wk_write, Wv_write, Wq_read, out_proj, _alibi_slopes, _alibi_strength_param, _content_read_gamma_raw, slot_in, slot_q, slot_k, slot_v, slot_out, _slotspace_gate_raw, RoPE buffers, etc.).
End state must include:
installable package (pyproject.toml or requirements.txt)
three runnable entrypoints: smoke, train-mini, probe
minimal tests (shape, checkpoint-key stability, NaN smoke)
docs sufficient for a stranger to run
Work in one session. Do not ask me questions; make reasonable defaults and document them in docs/ASSUMPTIONS.md.
Required outputs (deliverables checklist)
Create / update these at repo root:
README.md (public intro + quickstart commands)
LICENSE (use MIT unless notebooks indicate a different preference)
CITATION.cff (fill with placeholder author/project name; user can edit)
CHANGELOG.md (start with v0.1.0 “initial public release candidate”)
.gitignore
pyproject.toml (preferred) or requirements.txt
REPRODUCIBILITY.md (what to run, expected outputs, seed notes)
docs/ folder with:
docs/METHOD.md (ASA/ASM explanation)
docs/EXPERIMENTS.md (inventory of experiments found in notebooks)
docs/ROADMAP.md (next steps)
docs/ASSUMPTIONS.md (defaults chosen without user input)
src/asa/ (or src/asm/) with:
asa.py (primitive module)
asm_block.py
asm_lm.py
config.py (dataclasses)
utils/ (tokenization helpers, logging, seeding)
scripts/ with:
smoke.py
train_mini.py
probe_paris_margin.py (or similar; derived from notebooks)
tests/ with:
test_shapes.py
test_checkpoint_keys.py
test_no_nans_smoke.py
.github/workflows/ci.yml to run tests on CPU (fast)
Phase plan (execute in order)
Phase A — Inventory the notebooks (do not refactor yet)
Recursively scan /building_blocks/ for notebooks (*.ipynb) and python scripts.
Build an experiment inventory table with columns:
file path
purpose (training / probing / plotting / finetune / interventions / ablations)
key modules defined (ASA, ASMBlock, ASMLanguageModel, probes, utilities)
key metrics used (e.g., “Paris-margin”, routing EVR, slope, etc.)
artifacts produced (plots, checkpoints)
Write this into docs/EXPERIMENTS.md.
Identify the canonical definitions for:
ASA “cleaned + safer; checkpoint-stable”
ASA “training-focused efficient version”
ASA “refine-geometry logging + intervention”
ASM config / block / LM Capture differences in a short docs/METHOD.md section “Variants.”
Phase B — Extract the core library (public-facing)
Implement src/asa/asa.py using the checkpoint-stable “cleaned + safer” ASA as the default.
Implement optional variants behind flags or separate classes:
AddressedStateAttention (default, checkpoint-stable)
AddressedStateAttentionOnline (training-efficient online slotspace scan)
AddressedStateAttentionIntervene (geometry logging + refine interventions) Keep names stable and avoid breaking checkpoint parameter names. If variants require extras, add them without renaming shared parameters.
Implement ASMTrainConfig, ASMBlock, ASMLanguageModel, and build_model_from_cfg in src/asa/ files.
Provide a small utils/seed.py and utils/tokenization.py and utils/device.py.
Ensure forward signatures match notebook assumptions, including:
return_info
routing controls (routing_mode, routing_topk, read_weights_override, noise)
slot masking controls (slot_mask, slot_mask_where, slot_mask_scope) where present
Add lightweight typing + docstrings. Avoid heavy refactors.
Phase C — Create runnable scripts (the three entrypoints)
scripts/smoke.py
instantiate a tiny config
forward pass
one optimizer step on dummy data
assert no NaNs and expected shapes
scripts/train_mini.py
minimal dataset option: fallback to a tiny local text sample if HF datasets not available
if HF datasets used, guard imports and document install extras
run ~200–500 steps
print loss and save checkpoint under runs/
scripts/probe_paris_margin.py
implement the “Paris-margin” probe and the routing-vs-residual story as seen in notebooks
output one plot and a JSON summary to runs/probes/
ensure script runs on CPU, slower is acceptable
Phase D — Tests + CI
Add tests:
test_shapes.py: multiple toggles (slotspace on/off, content_read on/off) shapes invariant
test_checkpoint_keys.py: instantiate model, get state_dict() keys, compare against a stored golden list in tests/golden_state_dict_keys.txt (generate it once now)
test_no_nans_smoke.py: run a short forward, assert all finite
Add CI workflow that runs pytest -q on Python 3.10–3.12 CPU.
Phase E — Docs + release polish
Write README.md that includes:
8–12 line project overview
quickstart install
commands to run smoke/train/probe
“What is ASA” in 5 bullets
“Status: research code”
Write REPRODUCIBILITY.md:
exact commands
expected runtime class and expected output files
deterministic notes
Add CITATION.cff with placeholders.
Add LICENSE (MIT).
Add CHANGELOG.md with v0.1.0.
Add building_blocks/README.md explaining it’s raw archive.
Operating constraints and style rules
Prefer clarity over cleverness.
Avoid heavy dependencies; keep requirements minimal.
No hidden downloads required to run smoke test.
Every script must accept --device, --seed, --outdir.
Any assumptions go in docs/ASSUMPTIONS.md.
Do not delete anything. Move nothing out of /building_blocks/.
Make the repo “release candidate” ready at the end.
Final step
At the end, produce a single “Release Checklist” section in README.md with what is complete and what is left.
