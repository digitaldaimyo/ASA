"""Probe script that computes a Paris-margin style metric."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import torch

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig
from asa.utils.device import resolve_device
from asa.utils.seed import seed_everything
from asa.utils.tokenization import SimpleTokenizer


PROMPTS = [
    "The capital of France is ",
    "France's capital city is ",
    "Paris is the capital of ",
    "In Europe, the capital of France is ",
]

TARGETS = ["Paris", "London"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the Paris margin metric.")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--outdir", default="runs/probes")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    tokenizer = SimpleTokenizer.from_texts(PROMPTS + TARGETS)
    cfg = ASMTrainConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_slots=8,
        max_seq_len=64,
        dropout=0.0,
        use_content_read=True,
    )
    model = build_model_from_cfg(cfg).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()

    margins: List[float] = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits, infos = model(input_ids, return_info=True)
        last_logits = logits[0, -1]
        paris_id = tokenizer.encode("Paris")[0]
        london_id = tokenizer.encode("London")[0]
        margin = (last_logits[paris_id] - last_logits[london_id]).item()
        margins.append(margin)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(margins)), margins)
    ax.set_title("Paris-margin probe")
    ax.set_ylabel("logit(Paris) - logit(London)")
    ax.set_xlabel("prompt index")
    fig.tight_layout()
    plot_path = os.path.join(args.outdir, "paris_margin.png")
    fig.savefig(plot_path)

    summary = {
        "margins": margins,
        "mean_margin": sum(margins) / len(margins),
    }
    summary_path = os.path.join(args.outdir, "paris_margin.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved", plot_path)
    print("Saved", summary_path)


if __name__ == "__main__":
    main()
