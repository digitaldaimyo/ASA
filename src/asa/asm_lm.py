"""ASM language model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from asa.asa import (
    AddressedStateAttention,
    AddressedStateAttentionIntervene,
    AddressedStateAttentionOnline,
)
from asa.asm_block import ASMBlock
from asa.config import ASMTrainConfig


class ASMLanguageModel(nn.Module):
    def __init__(self, cfg: ASMTrainConfig, attn_class: Type[nn.Module]) -> None:
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
                    attn_class=attn_class,
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
                    min_strength=cfg.min_strength,
                    use_content_read=cfg.use_content_read,
                    content_read_init=cfg.content_read_init,
                    content_read_max_gamma=cfg.content_read_max_gamma,
                    slotspace_dropout=cfg.slotspace_dropout,
                    slotspace_signed_weights=cfg.slotspace_signed_weights,
                    use_rope_slotspace=cfg.use_rope_slotspace,
                    rope_base_slotspace=cfg.rope_base_slotspace,
                    write_chunk_size=cfg.write_chunk_size,
                    slotspace_dim=cfg.slotspace_dim,
                    slotspace_chunk_size=cfg.slotspace_chunk_size,
                    use_slotspace_refine=cfg.use_slotspace_refine,
                    slotspace_gate_init=cfg.slotspace_gate_init,
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
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_info: bool = False,
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "all",
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
                attention_mask=attention_mask,
                return_info=return_info,
                routing_mode=routing_mode,
                routing_topk=routing_topk,
                read_weights_override=read_weights_override,
                routing_noise=routing_noise,
                routing_noise_scale=routing_noise_scale,
                slot_mask=slot_mask,
                slot_mask_where=slot_mask_where,
                slot_mask_scope=slot_mask_scope,
                **asa_kwargs,
            )
            if return_info and infos is not None:
                infos.append(info or {})

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, infos


def _resolve_attn_class(variant: str) -> Type[nn.Module]:
    normalized = variant.strip().lower()
    if normalized == "baseline":
        return AddressedStateAttention
    if normalized == "online":
        return AddressedStateAttentionOnline
    if normalized == "intervene":
        return AddressedStateAttentionIntervene
    raise ValueError(f"Unknown ASA variant: {variant}")


def build_model_from_cfg(cfg: ASMTrainConfig, variant: str = "baseline") -> ASMLanguageModel:
    return ASMLanguageModel(cfg, attn_class=_resolve_attn_class(variant))
