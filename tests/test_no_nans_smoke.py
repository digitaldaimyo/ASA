import torch

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig


def test_no_nans_smoke():
    cfg = ASMTrainConfig(
        vocab_size=64,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        num_slots=8,
        max_seq_len=16,
        dropout=0.0,
        use_content_read=True,
    )
    model = build_model_from_cfg(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(inputs)
    assert torch.isfinite(logits).all()
