"""Configuration dataclasses for ASM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ASMTrainConfig:
    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"

    max_seq_len: int = 256
    stride_frac_val: float = 0.50
    seed: int = 1337

    # Sample budgets
    train_samples_target: int = 100_000
    val_samples_target: int = 5_000

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1_000
    total_steps: int = 75_000
    eval_interval: int = 1_000
    log_interval: int = 100

    # Model
    vocab_size: int = 50257
    embed_dim: int = 384
    num_layers: int = 6
    num_heads: int = 8
    num_slots: int = 32
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    tie_weights: bool = True

    # Addressed State Attention (ASA) / numerics
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    slot_dropout: float = 0.05
    state_fp32: bool = True
    normalize_k: bool = False

    # Positions
    use_abs_pos: bool = False
    use_rope_keys: bool = True
    rope_base: float = 10000.0
    use_alibi_write: bool = True
    alibi_strength_init: float = 0.1
    learn_alibi_strength: bool = True

    # Content read / slotspace refine
    use_content_read: bool = False
    content_read_init: float = -2.0
    content_read_max_gamma: float = 3.0
    write_chunk_size: int = 0
    slotspace_dim: int = 64
    slotspace_chunk_size: int = 0
    use_slotspace_refine: bool = False
    slotspace_gate_init: float = -3.0
    slotspace_gate_max: float = 1.0

    # Misc
    enable_compiled: bool = False
    tag: str = "asa_run"
    cache_dir: str = "runs/cache"
    val_windows_cache: str = "runs/val_windows_cache.pkl"
    resume_path: str = ""
    micro_batch_size: int = 0
    grad_accum_steps: int = 1
