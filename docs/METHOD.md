# Method Overview

## Addressed State Attention (ASA)

ASA routes token information into a small set of persistent slots. Each token
produces write keys/values, which are aggregated into slot states; read queries
then attend over slot states and project back into the token stream. The public
implementation in `src/asa/asa.py` preserves checkpoint-critical parameter names
from the notebooks while providing a simplified, runnable baseline.

Core concepts:
- **Slots** are learned keys that receive writes and serve reads.
- **Write path** aggregates token information into slot states.
- **Read path** queries slot states back into the sequence.
- Optional **content read** attends over token keys directly.
- Optional **slotspace refinement** runs a slot-to-slot attention pass.

## Addressed State Models (ASM)

ASM composes ASA blocks into a language model by stacking `ASMBlock` and adding
an embedding layer plus a prediction head. Configuration mirrors the notebooks
for compatibility but defaults to smaller sizes for quick runs.

## Variants

The notebooks define three canonical ASA variants:

1. **Cleaned + safer (checkpoint-stable)**
   - Emphasizes naming stability and clearer control flow.
   - Preserves parameters like `slot_keys`, `Wk_write`, `Wv_write`, `Wq_read`,
     `out_proj`, `_alibi_slopes`, `_alibi_strength_param`,
     `_content_read_gamma_raw`, and `_slotspace_gate_raw`.

2. **Training-focused efficient (online slotspace scan)**
   - Computes slotspace refinement within the write/read scan to save memory.
   - Avoids storing large read-weight buffers.

3. **Refine-geometry logging + intervention**
   - Captures routing logits and uses interventions to build energy basins
     (e.g., Paris-margin probes).
   - Includes refine-delta decomposition into parallel/orthogonal components
     for logging and intervention gating.

The library exposes these variants as:
- `AddressedStateAttention` (default, checkpoint-stable)
- `AddressedStateAttentionOnline` (training-efficient)
- `AddressedStateAttentionIntervene` (returns routing info)
