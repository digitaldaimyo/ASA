from pathlib import Path

from asa.asm_lm import build_model_from_cfg
from asa.config import ASMTrainConfig


def test_checkpoint_keys_match_golden():
    cfg = ASMTrainConfig(
        vocab_size=32,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        num_slots=8,
        max_seq_len=8,
    )
    model = build_model_from_cfg(cfg)
    keys = sorted(model.state_dict().keys())
    golden_path = Path(__file__).parent / "golden_state_dict_keys.txt"
    golden = [line.strip() for line in golden_path.read_text().splitlines() if line.strip()]
    assert keys == golden
