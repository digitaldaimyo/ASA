"""Smoke test entrypoint for ASA/ASM."""

from __future__ import annotations

import argparse
import os

import torch

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig
from asa.utils.device import resolve_device
from asa.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny ASA/ASM smoke test.")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--outdir", default="runs/smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = ASMTrainConfig(
        vocab_size=128,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_slots=8,
        max_seq_len=32,
        dropout=0.0,
        use_content_read=True,
        use_slotspace_refine=True,
    )
    model = build_model_from_cfg(cfg).to(device)
    model.train()

    inputs = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    logits, _ = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size), inputs.view(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()

    if torch.isnan(loss).any():
        raise RuntimeError("NaN detected in loss")

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.outdir, "smoke.pt"))
    print("Smoke test complete. Loss:", loss.item())


if __name__ == "__main__":
    main()
