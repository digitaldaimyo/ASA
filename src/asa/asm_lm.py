"""ASM language model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from asa.asm_block import ASMBlock
from asa.config import ASMTrainConfig


class ASMLanguageModel(nn.Module):
    def __init__(self, cfg: ASMTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim) if cfg.use_abs_pos else None
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                ASMBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    num_slots=cfg.num_slots,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    read_temperature=cfg.read_temperature,
                    write_temperature=cfg.write_temperature,
                    slot_dropout=cfg.slot_dropout,
                    state_fp32=cfg.state_fp32,
                    normalize_k=cfg.normalize_k,
                    use_rope_keys=cfg.use_rope_keys,
                    rope_base=cfg.rope_base,
                    use_alibi_write=cfg.use_alibi_write,
                    alibi_strength_init=cfg.alibi_strength_init,
                    learn_alibi_strength=cfg.learn_alibi_strength,
                    use_content_read=cfg.use_content_read,
                    content_read_init=cfg.content_read_init,
                    content_read_max_gamma=cfg.content_read_max_gamma,
                    write_chunk_size=cfg.write_chunk_size,
                    slotspace_dim=cfg.slotspace_dim,
                    slotspace_chunk_size=cfg.slotspace_chunk_size,
                    use_slotspace_refine=cfg.use_slotspace_refine,
                    slotspace_gate_init=cfg.slotspace_gate_init,
                    slotspace_gate_max=cfg.slotspace_gate_max,
                    enable_compiled=cfg.enable_compiled,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_info: bool = False,
        info_level: str = "basic",
        info_cfg: Optional[Dict[str, bool]] = None,
        **asa_kwargs,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos_ids)[None, :, :]
        x = self.drop(x)

        infos: Optional[List[Dict[str, torch.Tensor]]] = [] if return_info else None
        for block in self.blocks:
            x, info = block(
                x,
                return_info=return_info,
                info_level=info_level,
                info_cfg=info_cfg,
                **asa_kwargs,
            )
            if return_info and infos is not None:
                infos.append(info or {})

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, infos


def build_model_from_cfg(cfg: ASMTrainConfig) -> ASMLanguageModel:
    return ASMLanguageModel(cfg)
