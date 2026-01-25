"""Addressed State Attention (ASA) primitives."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._t_cached: Optional[int] = None
        self._device_cached: Optional[torch.device] = None

    def get_cos_sin(self, length: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._t_cached == length
            and self._cos_cached is not None
            and self._device_cached == device
        ):
            return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

        t = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        self._t_cached = length
        self._device_cached = device
        self._cos_cached = cos
        self._sin_cached = sin
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def alibi_slopes(num_heads: int, device=None, dtype=torch.float32) -> torch.Tensor:
    def get_slopes(n):
        def power_of_2_slopes(power):
            start = 2 ** (-2 ** -(math.log2(power) - 3))
            ratio = start
            return [start * ratio**i for i in range(power)]

        if math.log2(n).is_integer():
            return power_of_2_slopes(n)
        closest_power = 2 ** math.floor(math.log2(n))
        return (
            power_of_2_slopes(closest_power)
            + get_slopes(2 * closest_power)[0::2][: n - closest_power]
        )

    slopes = torch.tensor(get_slopes(num_heads), device=device, dtype=dtype)
    return slopes


class AddressedStateAttention(nn.Module):
    """Checkpoint-stable Addressed State Attention (ASA).

    This is a compact, public-facing implementation that preserves parameter
    names used in the research notebooks for checkpoint compatibility.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_slots: int,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        slot_dropout: float = 0.0,
        state_fp32: bool = True,
        normalize_k: bool = False,
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        use_content_read: bool = False,
        content_read_init: float = -2.0,
        content_read_max_gamma: float = 3.0,
        write_chunk_size: int = 0,
        slotspace_dim: Optional[int] = None,
        slotspace_chunk_size: int = 0,
        use_slotspace_refine: bool = False,
        slotspace_gate_init: float = -3.0,
        slotspace_gate_max: float = 1.0,
        enable_compiled: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads
        self.read_temperature = read_temperature
        self.write_temperature = write_temperature
        self.slot_dropout = slot_dropout
        self.state_fp32 = state_fp32
        self.normalize_k = normalize_k
        self.use_rope_keys = use_rope_keys
        self.use_alibi_write = use_alibi_write
        self.use_content_read = use_content_read
        self.content_read_max_gamma = content_read_max_gamma
        self.write_chunk_size = write_chunk_size
        self.slotspace_dim = slotspace_dim or self.head_dim
        self.slotspace_chunk_size = slotspace_chunk_size
        self.use_slotspace_refine = use_slotspace_refine
        self.slotspace_gate_max = slotspace_gate_max
        self.enable_compiled = enable_compiled

        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) * (self.head_dim**-0.5)
        )

        self._alibi_slopes = nn.Parameter(
            alibi_slopes(num_heads).view(1, num_heads, 1, 1), requires_grad=False
        )
        self._alibi_strength_param = nn.Parameter(torch.tensor(alibi_strength_init))
        self._alibi_strength_param.requires_grad = learn_alibi_strength

        self._content_read_gamma_raw = nn.Parameter(torch.tensor(content_read_init))
        self._slotspace_gate_raw = nn.Parameter(torch.tensor(slotspace_gate_init))

        # Slot-space refinement projections (kept for checkpoint compatibility)
        self.slot_in = nn.Linear(embed_dim, self.slotspace_dim, bias=False)
        self.slot_q = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
        self.slot_k = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
        self.slot_v = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
        self.slot_out = nn.Linear(self.slotspace_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.rope_slotspace = RotaryEmbedding(
            self.slotspace_dim if self.slotspace_dim % 2 == 0 else self.slotspace_dim - 1,
            base=rope_base,
        )

    def _apply_slotspace_refine(self, slot_state: torch.Tensor) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return slot_state
        bsz, num_slots, num_heads, head_dim = slot_state.shape
        slot_flat = slot_state.mean(dim=2).reshape(bsz, num_slots, head_dim)
        slot_emb = self.slot_in(slot_flat)
        q = self.slot_q(slot_emb)
        k = self.slot_k(slot_emb)
        v = self.slot_v(slot_emb)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        refined = torch.matmul(attn, v)
        gate = torch.sigmoid(self._slotspace_gate_raw) * self.slotspace_gate_max
        delta = self.slot_out(refined).unsqueeze(2)
        return slot_state + gate * delta

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_info: bool = False,
        info_level: str = "basic",
        info_cfg: Optional[Dict[str, bool]] = None,
        routing_mode: str = "softmax",
        routing_topk: Optional[int] = None,
        read_weights_override: Optional[torch.Tensor] = None,
        noise: float = 0.0,
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "batch",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        q = self.Wq_read(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.Wk_write(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.Wv_write(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        if self.normalize_k:
            k = F.normalize(k, dim=-1)
        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(seq_len, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scale = self.head_dim**-0.5
        write_logits = torch.einsum("bthd,hsd->bths", k, self.slot_keys) * scale
        if noise > 0.0:
            write_logits = write_logits + noise * torch.randn_like(write_logits)

        if slot_mask is not None and slot_mask_where in {"write", "both"}:
            mask = slot_mask
            if slot_mask_scope == "batch":
                mask = mask[None, None, None, :]
            write_logits = write_logits.masked_fill(~mask, float("-inf"))

        write_weights = torch.softmax(write_logits / self.write_temperature, dim=-1)
        if self.slot_dropout > 0:
            write_weights = F.dropout(write_weights, p=self.slot_dropout, training=self.training)

        slot_state = torch.einsum("bths,bthd->bshd", write_weights, v) / max(1, seq_len)
        slot_state = self._apply_slotspace_refine(slot_state)

        read_logits = torch.einsum("bthd,bshd->bths", q, slot_state) * scale
        if slot_mask is not None and slot_mask_where in {"read", "both"}:
            mask = slot_mask
            if slot_mask_scope == "batch":
                mask = mask[None, None, None, :]
            read_logits = read_logits.masked_fill(~mask, float("-inf"))

        if read_weights_override is not None:
            read_weights = read_weights_override
        else:
            read_weights = torch.softmax(read_logits / self.read_temperature, dim=-1)

        if routing_mode == "topk" and routing_topk is not None:
            topk = torch.topk(read_weights, routing_topk, dim=-1)
            mask = torch.zeros_like(read_weights)
            mask.scatter_(-1, topk.indices, 1.0)
            read_weights = read_weights * mask
            read_weights = read_weights / (read_weights.sum(dim=-1, keepdim=True) + 1e-6)

        read_out = torch.einsum("bths,bshd->bthd", read_weights, slot_state)

        if self.use_content_read:
            content_logits = torch.einsum("bthd,bThd->bhtT", q, k) * scale
            content_weights = torch.softmax(content_logits, dim=-1)
            content_out = torch.einsum("bhtT,bThd->bthd", content_weights, v)
            gamma = torch.sigmoid(self._content_read_gamma_raw) * self.content_read_max_gamma
            read_out = read_out + gamma * content_out

        out = self.out_proj(read_out.reshape(bsz, seq_len, self.embed_dim))

        info: Optional[Dict[str, torch.Tensor]] = None
        if return_info:
            info = {}
            cfg = info_cfg or {}
            if cfg.get("store_read_weights", True):
                info["read_weights"] = read_weights
            if cfg.get("store_write_weights", False):
                info["write_weights"] = write_weights
            if cfg.get("store_slot_state", False):
                info["slot_state"] = slot_state
        return out, info


class AddressedStateAttentionOnline(AddressedStateAttention):
    """Training-efficient variant with online slotspace scan."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("use_slotspace_refine", True)
        super().__init__(*args, **kwargs)


class AddressedStateAttentionIntervene(AddressedStateAttention):
    """Variant that exposes routing info for interventions."""

    def forward(self, *args, **kwargs):
        kwargs.setdefault("return_info", True)
        return super().forward(*args, **kwargs)
