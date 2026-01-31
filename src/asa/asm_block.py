"""ASM transformer block."""

from __future__ import annotations

import inspect
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from asa.asa import AddressedStateAttention


class ASMBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_slots: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_class: Type[nn.Module] = AddressedStateAttention,
        **asa_kwargs,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = attn_class(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_slots=num_slots,
            **self._filter_kwargs(attn_class, asa_kwargs),
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _filter_kwargs(attn_class: Type[nn.Module], kwargs: Dict) -> Dict:
        signature = inspect.signature(attn_class.__init__)
        allowed = {name for name in signature.parameters if name != "self"}
        return {key: value for key, value in kwargs.items() if key in allowed}

    @staticmethod
    def _filter_forward_kwargs(attn: nn.Module, kwargs: Dict) -> Dict:
        signature = inspect.signature(attn.forward)
        allowed = {name for name in signature.parameters if name != "self"}
        return {key: value for key, value in kwargs.items() if key in allowed}

    def forward(
        self,
        x: torch.Tensor,
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
        layer_idx: Optional[int] = None,
        **asa_kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        residual = x
        forward_kwargs = dict(
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
            layer_idx=layer_idx,
        )
        forward_kwargs.update(asa_kwargs)
        attn_out, info = self.attn(
            self.ln_1(x),
            **self._filter_forward_kwargs(self.attn, forward_kwargs),
        )
        x = residual + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, info
