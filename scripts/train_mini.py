"""Train a tiny ASM model on a local text sample."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig
from asa.utils.device import resolve_device
from asa.utils.seed import seed_everything
from asa.utils.tokenization import SimpleTokenizer


SAMPLE_TEXT = """
Addressed State Attention (ASA) is a research primitive for routing information
through a small set of learned slots. Addressed State Models (ASM) build on ASA
by composing blocks into a language model.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny ASM model.")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--outdir", default="runs/train_mini")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=64)
    return parser.parse_args()


def batchify(tokens: List[int], seq_len: int, batch_size: int, pad_id: int) -> torch.Tensor:
    if len(tokens) < seq_len:
        repeats = (seq_len // max(1, len(tokens))) + 1
        tokens = (tokens * repeats)[:seq_len]
    chunks = max(1, len(tokens) // seq_len)
    tokens = tokens[: chunks * seq_len]
    if len(tokens) < chunks * seq_len:
        tokens = tokens + [pad_id] * (chunks * seq_len - len(tokens))
    data = torch.tensor(tokens, dtype=torch.long).view(chunks, seq_len)
    idx = torch.randint(0, data.shape[0], (batch_size,))
    return data[idx]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    tokenizer = SimpleTokenizer.from_texts([SAMPLE_TEXT])
    tokens = tokenizer.encode(SAMPLE_TEXT)

    cfg = ASMTrainConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        num_slots=8,
        max_seq_len=args.seq_len,
        dropout=0.1,
        use_content_read=True,
        use_slotspace_refine=True,
    )
    model = build_model_from_cfg(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        batch = batchify(tokens, args.seq_len, batch_size=4, pad_id=tokenizer.vocab["<pad>"]).to(device)
        logits, _ = model(batch)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), batch.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    ckpt_path = os.path.join(args.outdir, "train_mini.pt")
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__, "tokenizer": tokenizer.vocab},
        ckpt_path,
    )
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"loss_mean": sum(losses) / len(losses)}, f, indent=2)
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
