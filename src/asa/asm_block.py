"""ASM transformer block."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        **asa_kwargs,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = AddressedStateAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_slots=num_slots,
            **asa_kwargs,
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_info: bool = False,
        info_level: str = "basic",
        info_cfg: Optional[Dict[str, bool]] = None,
        **asa_kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        residual = x
        attn_out, info = self.attn(
            self.ln_1(x),
            return_info=return_info,
            info_level=info_level,
            info_cfg=info_cfg,
            **asa_kwargs,
        )
        x = residual + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, info
