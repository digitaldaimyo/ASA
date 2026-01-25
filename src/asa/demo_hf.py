"""Hugging Face checkpoint demo for ASA/ASM."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from typing import Any, Dict, Tuple

import torch
from huggingface_hub import hf_hub_download

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig


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


def model_cfg_from_train_cfg(raw: Dict[str, Any]) -> ASMTrainConfig:
    norm = _normalize_cfg(raw)
    allowed = {f.name for f in fields(ASMTrainConfig)}
    filtered = {k: v for k, v in norm.items() if k in allowed}
    return ASMTrainConfig(**filtered)


def extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt, "root"
    raise ValueError("Unsupported checkpoint format for state_dict extraction")


def load_checkpoint(repo: str, ckpt_name: str | None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], str]:
    if ckpt_name is None:
        ckpt_name = "ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt"
    ckpt_path = hf_hub_download(repo_id=repo, filename=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = {}
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = ckpt["cfg"]

    state_dict, key = extract_state_dict(ckpt)
    return state_dict, cfg, key


def load_config(repo: str) -> Dict[str, Any]:
    try:
        cfg_path = hf_hub_download(repo_id=repo, filename="config.json")
        return _load_json(cfg_path)
    except Exception:
        return {}


def build_model(repo: str, cfg_from_ckpt: Dict[str, Any]) -> ASMTrainConfig:
    cfg_data = load_config(repo)
    if not cfg_data:
        cfg_data = cfg_from_ckpt
    cfg = model_cfg_from_train_cfg(cfg_data)
    return cfg


def run_demo(repo: str, ckpt: str | None, seq_len: int, gen_tokens: int) -> None:
    state_dict, cfg_from_ckpt, key = load_checkpoint(repo, ckpt)
    print(f"Checkpoint state dict source: {key}")
    cfg = build_model(repo, cfg_from_ckpt)
    model = build_model_from_cfg(cfg)

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint with strict=True")
    except RuntimeError as exc:
        print(f"Strict load failed: {exc}\nRetrying with strict=False")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model.eval()
    vocab = cfg.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq_len))
    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (1, seq_len, vocab), f"logits shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"
    print("PASS: forward logits shape + finite")

    if gen_tokens > 0:
        tokens = input_ids.clone()
        for _ in range(gen_tokens):
            with torch.no_grad():
                out, _ = model(tokens)
            next_id = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=1)
        print(f"Generated {gen_tokens} tokens. Final shape: {tuple(tokens.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF demo for ASA/ASM.")
    parser.add_argument("--repo", default="DigitalShogun/ASA-ASM-wikitext103-raw")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint filename")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--gen-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading from {args.repo}")
    run_demo(args.repo, args.ckpt, args.seq_len, args.gen_tokens)


if __name__ == "__main__":
    main()
