"""
Utilities for loading ASA/ASM checkpoints from Hugging Face.

Supports:
- legacy flat layout:
    config.json
    <ckpt>.pt
- new structured layout:
    checkpoints/<dataset>/<ckpt>.pt
    configs/<dataset>/<ckpt>.config.json
    metadata/<dataset>/<ckpt>.metadata.json (optional; not required for loading)

Also supports loading from a local path by passing repo_id=None and filename=<local path>.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any, Dict, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download

from asa.asm_lm import build_model_from_cfg
from asa.checkpoints import load_model_weights
from asa.config import ASMTrainConfig

DEFAULT_CKPT = (
    "ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt"
)

# ----------------------------
# Helpers: JSON + config mapping
# ----------------------------

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
    """
    Returns (state_dict, source_key)
    """
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        if all(isinstance(k, str) for k in ckpt.keys()):
            # Sometimes the entire checkpoint *is* a state_dict
            # (rare, but allow it).
            return ckpt, "root"
    raise ValueError("Unsupported checkpoint format for state_dict extraction")


# ----------------------------
# New: map ckpt path -> config path for structured repos
# ----------------------------

def infer_config_filename_from_ckpt(
    ckpt_filename: str,
    *,
    checkpoints_root: str = "checkpoints",
    configs_root: str = "configs",
    config_suffix: str = ".config.json",
) -> str:
    """
    Map:
      checkpoints/<dataset>/<name>.pt  ->  configs/<dataset>/<name>.config.json

    Works for nested ckpt_filename paths; uses POSIX semantics for HF paths.
    """
    p = PurePosixPath(ckpt_filename)
    parts = p.parts

    if checkpoints_root in parts:
        i = parts.index(checkpoints_root)
        if i + 1 >= len(parts):
            raise ValueError(f"Expected '{checkpoints_root}/<dataset>/...'; got: {ckpt_filename}")
        dataset = parts[i + 1]
        base = p.name[:-3] if p.name.endswith(".pt") else p.name
        return str(PurePosixPath(configs_root) / dataset / (base + config_suffix))

    # If it isn't under checkpoints/, don't guess—return conventional root config name.
    # Callers can pass config_filename explicitly if needed.
    return "config.json"


def infer_metadata_filename_from_ckpt(
    ckpt_filename: str,
    *,
    checkpoints_root: str = "checkpoints",
    metadata_root: str = "metadata",
    metadata_suffix: str = ".metadata.json",
) -> str:
    p = PurePosixPath(ckpt_filename)
    parts = p.parts
    if checkpoints_root in parts:
        i = parts.index(checkpoints_root)
        dataset = parts[i + 1]
        base = p.name[:-3] if p.name.endswith(".pt") else p.name
        return str(PurePosixPath(metadata_root) / dataset / (base + metadata_suffix))
    return "metadata.json"


# ----------------------------
# Loading: config resolution
# ----------------------------

def _try_hf_download_json(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Try to download filename from HF, parse as JSON, return dict or None.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return None

    try:
        data = _load_json(path)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _resolve_cfg_data(
    repo_id: Optional[str],
    *,
    ckpt: Any,
    ckpt_filename: str,
    config_filename: Optional[str],
) -> Dict[str, Any]:
    """
    Resolve cfg dict from:
    1) explicit config_filename (HF file if repo_id provided, else local file)
    2) inferred structured config path from ckpt_filename (HF file)
    3) root config.json (HF file)
    4) ckpt["cfg"] (if present)
    """
    # 1) explicit config file
    if config_filename is not None:
        if repo_id is None:
            # local file
            data = _load_json(config_filename)
            if not isinstance(data, dict):
                raise ValueError(f"config_filename must be a JSON object dict; got {type(data)}")
            return data
        else:
            data = _try_hf_download_json(repo_id, config_filename)
            if data:
                return data

    # 2) infer config from ckpt_filename (structured repos)
    if repo_id is not None:
        inferred = infer_config_filename_from_ckpt(ckpt_filename)
        if inferred and inferred != "config.json":
            data = _try_hf_download_json(repo_id, inferred)
            if data:
                return data

    # 3) fallback root config.json
    if repo_id is not None:
        data = _try_hf_download_json(repo_id, "config.json")
        if data:
            return data

    # 4) last resort: cfg inside checkpoint dict
    if isinstance(ckpt, dict):
        cfg = ckpt.get("cfg", None)
        if isinstance(cfg, dict) and cfg:
            return cfg

    raise ValueError(
        "No config found. Provide config_filename, or upload configs/<dataset>/<ckpt>.config.json "
        "or root config.json, or ensure ckpt contains ckpt['cfg']."
    )


# ----------------------------
# Public API
# ----------------------------

def load_pretrained(
    repo_id: Optional[str],
    filename: Optional[str] = None,
    *,
    variant: str = "baseline",
    device: Union[str, torch.device] = "cpu",
    config_filename: Optional[str] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any], ASMTrainConfig]:
    """
    Load a pretrained ASA/ASM model.

    Args:
      repo_id:
        - HF repo id (e.g. "DigitalShogun/ASA-ASM-wikitext103-raw")
        - OR None to load from a local checkpoint path.
      filename:
        - HF filename within the repo (can include subfolders),
          e.g. "checkpoints/fineweb/...pt"
        - OR local path to a checkpoint when repo_id=None.
      variant:
        "baseline" | "online" | "intervene"
      config_filename:
        If provided:
          - when repo_id is HF: treated as HF path inside repo
          - when repo_id is None: treated as local path
        If not provided, we infer configs/<dataset>/<ckpt>.config.json for structured repos.
      strict:
        passed to load_model_weights (checkpoint key allowlists apply there)

    Returns:
      (model, report, cfg_obj)
    """
    ckpt_name = filename or DEFAULT_CKPT

    # Load checkpoint bytes
    if repo_id is None:
        ckpt = torch.load(ckpt_name, map_location="cpu")
        ckpt_source = "local"
    else:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_source = "hf"

    state_dict, state_key = _extract_state_dict(ckpt)

    cfg_data = _resolve_cfg_data(
        repo_id,
        ckpt=ckpt,
        ckpt_filename=ckpt_name,
        config_filename=config_filename,
    )
    cfg_obj = _model_cfg_from_train_cfg(cfg_data)

    model = build_model_from_cfg(cfg_obj, variant=variant)
    report = load_model_weights(model, state_dict, strict=strict)

    report["state_dict_source"] = state_key
    report["checkpoint"] = ckpt_name
    report["checkpoint_source"] = ckpt_source
    report["config_filename"] = (
        config_filename
        if config_filename is not None
        else infer_config_filename_from_ckpt(ckpt_name)
    )

    model.to(device)
    model.eval()
    return model, report, cfg_obj
