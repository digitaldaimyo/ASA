"""Utilities for loading ASA/ASM checkpoints from Hugging Face."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, Dict, Tuple

import torch
from huggingface_hub import hf_hub_download

from asa.asm_lm import build_model_from_cfg
from asa.checkpoints import load_model_weights
from asa.config import ASMTrainConfig

DEFAULT_CKPT = (
    "ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt"
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    mapped = dict(raw)
    aliases = {
        "n_layer": "num_layers",
        "n_head": "num_heads",
        "n_embd": "embed_dim",
        "n_ctx": "max_seq_len",
        "n_positions": "max_seq_len",
        "n_slots": "num_slots",
        "slotspace_size": "slotspace_dim",
        "slotspace": "slotspace_dim",
    }
    for src, dst in aliases.items():
        if src in mapped and dst not in mapped:
            mapped[dst] = mapped[src]
    return mapped


def _model_cfg_from_train_cfg(raw: Dict[str, Any]) -> ASMTrainConfig:
    norm = _normalize_cfg(raw)
    allowed = {f.name for f in fields(ASMTrainConfig)}
    filtered = {k: v for k, v in norm.items() if k in allowed}
    return ASMTrainConfig(**filtered)


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt, "root"
    raise ValueError("Unsupported checkpoint format for state_dict extraction")


def _load_config(repo_id: str) -> Dict[str, Any]:
    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        return _load_json(cfg_path)
    except Exception:
        return {}


def load_pretrained(
    repo_id: str,
    filename: str | None = None,
    *,
    variant: str = "baseline",
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any], ASMTrainConfig]:
    ckpt_name = filename or DEFAULT_CKPT
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict, state_key = _extract_state_dict(ckpt)
    cfg_data = _load_config(repo_id)
    if not cfg_data and isinstance(ckpt, dict):
        cfg_data = ckpt.get("cfg", {})
    if not cfg_data:
        raise ValueError("No config found in checkpoint or config.json")

    cfg_obj = _model_cfg_from_train_cfg(cfg_data)
    model = build_model_from_cfg(cfg_obj, variant=variant)
    report = load_model_weights(model, state_dict, strict=True)
    report["state_dict_source"] = state_key
    report["checkpoint"] = ckpt_name

    model.to(device)
    model.eval()
    return model, report, cfg_obj
