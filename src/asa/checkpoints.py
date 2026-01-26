"""Checkpoint compatibility utilities for ASA/ASM."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import torch

CANONICAL_KEYS_PREFIXES = (
    "token_emb.weight",
    "ln_f.weight",
    "ln_f.bias",
    "lm_head.weight",
    "blocks.",
)

SAFE_MISSING_PATTERNS: Tuple[str, ...] = (
    r"^pos_emb\.weight$",
    r"^blocks\.\d+\.attn\._alibi_slopes$",
    r"^blocks\.\d+\.attn\._alibi_strength_param$",
    r"^blocks\.\d+\.attn\._content_read_gamma_raw$",
    r"^blocks\.\d+\.attn\._slotspace_gate_raw$",
    r"^blocks\.\d+\.attn\.slot_(in|q|k|v|out)\.(weight|bias)$",
    r"^blocks\.\d+\.mlp\.(0|3)\.bias$",
    r"^blocks\.\d+\.attn\.rope\.inv_freq$",
    r"^blocks\.\d+\.attn\.rope_slotspace\.inv_freq$",
)

SAFE_UNEXPECTED_PATTERNS: Tuple[str, ...] = SAFE_MISSING_PATTERNS
SAFE_MISMATCHED_PATTERNS: Tuple[str, ...] = SAFE_MISSING_PATTERNS


def detect_format(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("tok.") for key in keys):
        return "legacy_hf_v1"
    if any(key.startswith("norm.") for key in keys):
        return "legacy_hf_v1"
    if any(re.match(r"^blocks\.\d+\.asa\.", key) for key in keys):
        return "legacy_hf_v1"
    if any(key.startswith(CANONICAL_KEYS_PREFIXES) for key in keys):
        return "canonical"
    raise ValueError("Unknown checkpoint format; cannot detect naming scheme.")


def to_canonical(state_dict: Dict[str, torch.Tensor], format_id: str) -> Dict[str, torch.Tensor]:
    if format_id == "canonical":
        return dict(state_dict)
    if format_id != "legacy_hf_v1":
        raise ValueError(f"Unsupported checkpoint format: {format_id}")

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("tok."):
            new_key = new_key.replace("tok.", "token_emb.", 1)
        if new_key.startswith("norm."):
            new_key = new_key.replace("norm.", "ln_f.", 1)
        new_key = re.sub(r"^blocks\.(\d+)\.norm1\.", r"blocks.\1.ln_1.", new_key)
        new_key = re.sub(r"^blocks\.(\d+)\.norm2\.", r"blocks.\1.ln_2.", new_key)
        new_key = re.sub(r"^blocks\.(\d+)\.asa\.", r"blocks.\1.attn.", new_key)
        remapped[new_key] = value
    return remapped


def _partition_keys(keys: Iterable[str], patterns: Tuple[str, ...]) -> Tuple[List[str], List[str]]:
    allowed: List[str] = []
    disallowed: List[str] = []
    compiled = [re.compile(pat) for pat in patterns]
    for key in keys:
        if any(regex.search(key) for regex in compiled):
            allowed.append(key)
        else:
            disallowed.append(key)
    return allowed, disallowed


def load_model_weights(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> Dict[str, List[str] | str]:
    format_id = detect_format(state_dict)
    canonical = to_canonical(state_dict, format_id)
    model_state = model.state_dict()

    load_state: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    mismatched: List[str] = []

    for key, value in canonical.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            mismatched.append(key)
            continue
        load_state[key] = value

    missing = [key for key in model_state.keys() if key not in load_state]

    allowed_missing, disallowed_missing = _partition_keys(missing, SAFE_MISSING_PATTERNS)
    allowed_unexpected, disallowed_unexpected = _partition_keys(
        unexpected, SAFE_UNEXPECTED_PATTERNS
    )
    allowed_mismatched, disallowed_mismatched = _partition_keys(
        mismatched, SAFE_MISMATCHED_PATTERNS
    )

    if strict and (disallowed_missing or disallowed_unexpected or disallowed_mismatched):
        raise RuntimeError(
            "Checkpoint load blocked due to unallowlisted keys:\n"
            f"  Missing: {disallowed_missing}\n"
            f"  Unexpected: {disallowed_unexpected}\n"
            f"  Shape mismatched: {disallowed_mismatched}"
        )

    model.load_state_dict(load_state, strict=False)
    return {
        "format_id": format_id,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "allowed_missing": allowed_missing,
        "allowed_unexpected": allowed_unexpected,
        "allowed_mismatched": allowed_mismatched,
        "disallowed_missing": disallowed_missing,
        "disallowed_unexpected": disallowed_unexpected,
        "disallowed_mismatched": disallowed_mismatched,
    }
