"""Hugging Face checkpoint demo for ASA/ASM."""

from __future__ import annotations

import argparse
import torch

from asa.load_pretrained import load_pretrained


def run_demo(repo: str, ckpt: str | None, seq_len: int, gen_tokens: int, variant: str) -> None:
    model, report, cfg = load_pretrained(repo, ckpt, variant=variant, device="cpu")
    print(f"Checkpoint state dict source: {report['state_dict_source']}")
    print(
        "Allowlisted gaps - missing:",
        len(report["allowed_missing"]),
        "unexpected:",
        len(report["allowed_unexpected"]),
        "mismatched:",
        len(report["allowed_mismatched"]),
    )

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
    parser.add_argument("--variant", default="baseline", choices=["baseline", "online", "intervene"])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--gen-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading from {args.repo}")
    run_demo(args.repo, args.ckpt, args.seq_len, args.gen_tokens, args.variant)


if __name__ == "__main__":
    main()
