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
    if x.dim() >= 3 and cos.shape[-2] != x.shape[-2] and cos.shape[-2] == x.shape[-3]:
        cos = cos.transpose(-3, -2)
        sin = sin.transpose(-3, -2)
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


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def phi(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class AddressedStateAttention(nn.Module):
    """Addressed State Attention (ASA) reference implementation.

    This matches the "cleaned + safer; checkpoint-stable" notebook variant.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature)
        self.state_fp32 = bool(state_fp32)
        self.slot_dropout = float(slot_dropout)
        self.normalize_k = bool(normalize_k)
        self.routing_override = None

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.use_slotspace_refine = bool(use_slotspace_refine)
        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)

        self.write_chunk_size = int(write_chunk_size)
        self.slotspace_chunk_size = int(slotspace_chunk_size)

        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) / math.sqrt(self.head_dim)
        )

        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope_keys else None

        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(num_heads), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(num_heads), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        self.use_rope_slotspace = bool(use_rope_slotspace) and bool(self.use_slotspace_refine)
        if self.use_slotspace_refine:
            self.slot_in = nn.Linear(num_slots, self.slotspace_dim, bias=False)
            self.slot_q = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, num_slots, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))

            if self.use_rope_slotspace:
                if self.slotspace_dim % 2 != 0:
                    raise ValueError("use_rope_slotspace requires even slotspace_dim")
                self.rope_slotspace = RotaryEmbedding(
                    self.slotspace_dim, base=float(rope_base_slotspace)
                )
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = None
            self.slot_k = None
            self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(self.alibi_strength, dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    def _resolve_slot_mask(
        self,
        slot_mask: Optional[torch.Tensor],
        *,
        B: int,
        H: int,
        L: int,
        K: int,
        device,
        dtype,
        scope: str,
    ) -> Optional[torch.Tensor]:
        if slot_mask is None:
            slot_mask = getattr(self, "slot_mask", None)
        if slot_mask is None:
            return None

        sm = slot_mask.to(device=device, dtype=dtype)
        if sm.ndim != 1 or sm.numel() != K:
            raise ValueError(f"slot_mask must be shape [K]={K}, got {tuple(sm.shape)}")

        sm = sm.view(1, 1, 1, K)
        if scope == "all":
            return sm.expand(B, H, L, K)
        if scope == "last_pos_only":
            out = torch.ones((B, H, L, K), device=device, dtype=dtype)
            out[:, :, -1:, :] = sm.expand(B, H, 1, K)
            return out
        raise ValueError(f"Unknown slot_mask_scope={scope!r}")

    @staticmethod
    def _apply_hard_mask_and_renorm(w: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        w = w * keep.to(w.dtype)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _compute_read_weights(
        self,
        *,
        read_logits: torch.Tensor,
        read_logits_key: torch.Tensor,
        read_logits_content: Optional[torch.Tensor],
        routing_mode: str,
        routing_topk: int,
        read_weights_override: Optional[torch.Tensor],
        routing_noise: Optional[str],
        routing_noise_scale: float,
        rtemp: float,
        sm: Optional[torch.Tensor],
        slot_mask_where: str,
        B: int,
        H: int,
        L: int,
        K: int,
        T_total: int,
        t0: int,
        t1: int,
    ) -> torch.Tensor:
        if routing_noise is not None:
            if routing_noise == "gumbel":
                u = torch.rand_like(read_logits)
                g = -torch.log(-torch.log(u.clamp_min(1e-8)).clamp_min(1e-8))
                read_logits = read_logits + routing_noise_scale * g
            elif routing_noise == "gaussian":
                read_logits = read_logits + routing_noise_scale * torch.randn_like(read_logits)
            else:
                raise ValueError(f"Unknown routing_noise={routing_noise}")

        if self.routing_override is not None:
            if callable(self.routing_override):
                ctx = {
                    "t0": t0,
                    "t1": t1,
                    "B": B,
                    "H": H,
                    "T": T_total,
                    "K": K,
                    "d": self.head_dim,
                    "rtemp": rtemp,
                    "state_dtype": read_logits.dtype,
                    "q_read_c": None,
                    "slot_keys": self.slot_keys,
                    "slot_state_t": None,
                    "valid": None,
                    "slot_mask": None,
                    "slot_mask_where": slot_mask_where,
                }
                read_w = self.routing_override(
                    t0,
                    t1,
                    read_logits,
                    read_logits_key,
                    read_logits_content,
                    ctx,
                )
            else:
                read_w = self.routing_override[:, :, t0:t1, :].to(read_logits.dtype)

            read_w = torch.nan_to_num(read_w, nan=0.0, posinf=0.0, neginf=0.0)
            read_w = read_w.clamp_min(0.0)
            read_w = read_w / read_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            if routing_mode == "softmax":
                read_w = torch.softmax(read_logits / rtemp, dim=-1)
            elif routing_mode == "top1":
                top = read_logits.argmax(dim=-1)
                read_w = F.one_hot(top, num_classes=K).to(read_logits.dtype)
            elif routing_mode == "topk":
                kk = max(1, min(K, int(routing_topk)))
                vals, idx = torch.topk(read_logits, k=kk, dim=-1)
                masked = torch.full_like(read_logits, float("-inf"))
                masked.scatter_(-1, idx, vals)
                read_w = torch.softmax(masked / rtemp, dim=-1)
            elif routing_mode == "external":
                if read_weights_override is None:
                    raise ValueError("routing_mode='external' requires read_weights_override")
                if read_weights_override.shape[-2] == T_total:
                    read_w = read_weights_override[:, :, t0:t1, :]
                else:
                    read_w = read_weights_override
                read_w = read_w / read_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                raise ValueError(f"Unknown routing_mode={routing_mode}")

        if slot_mask_where == "read" and sm is not None:
            read_w = self._apply_hard_mask_and_renorm(read_w, (sm > 0.0))
        return read_w

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "all",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)
        q_read = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = (torch.rand((H, K), device=x.device) < self.slot_dropout)
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype)

        wtemp = max(1e-6, self.write_temperature)
        write_logits = write_logits / wtemp

        alibi_bias_applied = None
        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos_i = torch.arange(T, device=x.device, dtype=state_dtype)
            alibi_bias = slopes.view(1, H, 1, 1) * pos_i.view(1, 1, 1, T)
            write_logits = write_logits + alibi_bias
            alibi_bias_applied = alibi_bias

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        content_read_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)
        read_weights = torch.empty((B, H, T, K), device=x.device, dtype=q_read.dtype)

        slot_state_norm_t = (
            torch.empty((B, H, T, K), device=x.device, dtype=torch.float32) if return_info else None
        )

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        if return_info:
            read_logits_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_key_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_content_full = (
                torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
                if self.use_content_read
                else None
            )
        else:
            read_logits_full = None
            read_logits_key_full = None
            read_logits_content_full = None

        write_chunk = self.write_chunk_size or T
        for t0 in range(0, T, write_chunk):
            t1 = min(T, t0 + write_chunk)
            L = t1 - t0

            wlog_c = write_logits[:, :, :, t0:t1]

            m_c, _ = torch.cummax(wlog_c, dim=-1)
            m_new = torch.maximum(m_state.unsqueeze(-1), m_c)

            scale = torch.exp(m_state.unsqueeze(-1) - m_new)

            denom_c = denom_state.unsqueeze(-1) * scale
            numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

            w_new = self._safe_exp_sub_max(wlog_c, m_new)

            denom_c = denom_c + torch.cumsum(w_new, dim=-1)
            v_c = v_write[:, :, t0:t1, :].to(state_dtype)
            add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
            numer_c = numer_c + add

            slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
            slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

            q_read_c = q_read[:, :, t0:t1, :]
            read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)

            read_logits_content = None
            if self.use_content_read:
                read_logits_content = torch.einsum(
                    "bhld,bhlkd->bhlk",
                    q_read_c,
                    slot_state_t.to(q_read_c.dtype),
                ) / math.sqrt(d)

            sm = self._resolve_slot_mask(
                slot_mask,
                B=B,
                H=H,
                L=L,
                K=K,
                device=x.device,
                dtype=read_logits_key.dtype,
                scope=slot_mask_scope,
            )

            if slot_mask_where == "read":
                if sm is not None:
                    read_logits_key = read_logits_key.masked_fill(sm <= 0.0, float("-inf"))
                    if self.use_content_read and read_logits_content is not None:
                        read_logits_content = read_logits_content.masked_fill(sm <= 0.0, float("-inf"))
            elif slot_mask_where == "content_read_only":
                if sm is not None and self.use_content_read and read_logits_content is not None:
                    read_logits_content = read_logits_content.masked_fill(sm <= 0.0, 0.0)
            elif slot_mask_where == "slotspace_only":
                pass
            else:
                raise ValueError(f"Unknown slot_mask_where={slot_mask_where!r}")

            read_logits = read_logits_key
            if self.use_content_read and read_logits_content is not None:
                read_logits = read_logits + content_read_gamma.to(read_logits.dtype) * read_logits_content

            if return_info:
                read_logits_full[:, :, t0:t1, :] = read_logits.to(state_dtype)
                read_logits_key_full[:, :, t0:t1, :] = read_logits_key.to(state_dtype)
                if self.use_content_read and read_logits_content_full is not None:
                    read_logits_content_full[:, :, t0:t1, :] = read_logits_content.to(state_dtype)

            read_w_c = self._compute_read_weights(
                read_logits=read_logits,
                read_logits_key=read_logits_key,
                read_logits_content=read_logits_content,
                routing_mode=routing_mode,
                routing_topk=routing_topk,
                read_weights_override=read_weights_override,
                routing_noise=routing_noise,
                routing_noise_scale=routing_noise_scale,
                rtemp=rtemp,
                sm=sm,
                slot_mask_where=slot_mask_where,
                B=B,
                H=H,
                L=L,
                K=K,
                T_total=T,
                t0=t0,
                t1=t1,
            )

            read_weights[:, :, t0:t1, :] = read_w_c

            out_h[:, :, t0:t1, :] = torch.einsum(
                "bhlk,bhlkd->bhld",
                read_w_c.to(state_dtype),
                slot_state_t.to(state_dtype),
            )

            if return_info:
                slot_state_norm_t[:, :, t0:t1, :] = slot_state_t.to(torch.float32).norm(dim=-1)

            m_state = m_new[:, :, :, -1]
            denom_state = denom_c[:, :, :, -1]
            numer_state = numer_c[:, :, :, -1, :]

        slotspace_delta_norm_mean = None
        if self.use_slotspace_refine:
            slotspace_dtype = state_dtype
            M = self.slotspace_dim

            u = self.slot_in(read_weights.to(slotspace_dtype))
            q_s = self.slot_q(u)
            k_s = self.slot_k(u)
            v_s = self.slot_v(u)

            if self.use_rope_slotspace:
                cos_s, sin_s = self.rope_slotspace.get_cos_sin(T, device=x.device, dtype=q_s.dtype)
                q_s = apply_rope(q_s, cos_s, sin_s)
                k_s = apply_rope(k_s, cos_s, sin_s)

            qf = phi(q_s)
            kf = phi(k_s)

            if valid is not None:
                mask = valid.view(B, 1, T, 1).to(slotspace_dtype)
                qf = qf * mask
                kf = kf * mask
                v_s = v_s * mask

            u2 = torch.empty((B, H, T, M), device=x.device, dtype=slotspace_dtype)

            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)

            slotspace_chunk = self.slotspace_chunk_size or T
            for t0 in range(0, T, slotspace_chunk):
                t1 = min(T, t0 + slotspace_chunk)
                qf_c = qf[:, :, t0:t1, :]
                kf_c = kf[:, :, t0:t1, :]
                v_c = v_s[:, :, t0:t1, :]

                kv = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_c)
                S_c = torch.cumsum(kv, dim=2) + S_state.unsqueeze(2)
                Z_c = (torch.cumsum(kf_c, dim=2) + Z_state.unsqueeze(2)).clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1).clamp_min(1e-8)
                u2[:, :, t0:t1, :] = num / den

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

            u2 = self.slotspace_dropout(u2)

            slot_w = self.slot_out(u2)
            if self.slotspace_signed_weights:
                slot_w = torch.tanh(slot_w)
            else:
                slot_w = torch.softmax(slot_w, dim=-1)

            if slot_mask_where == "slotspace_only":
                sm_full = self._resolve_slot_mask(
                    slot_mask,
                    B=B,
                    H=H,
                    L=T,
                    K=K,
                    device=x.device,
                    dtype=slot_w.dtype,
                    scope=slot_mask_scope,
                )
                if sm_full is not None:
                    slot_w = slot_w * (sm_full > 0.0).to(slot_w.dtype)
                    if not self.slotspace_signed_weights:
                        slot_w = slot_w / slot_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)

            denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
            numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
            m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

            delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            delta_norm_count = 0

            for t0 in range(0, T, write_chunk):
                t1 = min(T, t0 + write_chunk)

                wlog_c = write_logits[:, :, :, t0:t1]

                m_c, _ = torch.cummax(wlog_c, dim=-1)
                m_new = torch.maximum(m_state.unsqueeze(-1), m_c)

                scale = torch.exp(m_state.unsqueeze(-1) - m_new)
                denom_c = denom_state.unsqueeze(-1) * scale
                numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

                w_new = self._safe_exp_sub_max(wlog_c, m_new)
                denom_c = denom_c + torch.cumsum(w_new, dim=-1)

                v_c = v_write[:, :, t0:t1, :].to(state_dtype)
                add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
                numer_c = numer_c + add

                slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
                slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

                slot_w_c = slot_w[:, :, t0:t1, :].to(state_dtype)
                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c, slot_state_t.to(state_dtype))

                out_h[:, :, t0:t1, :] = out_h[:, :, t0:t1, :] + gate * delta_c

                delta_norm_sum = delta_norm_sum + delta_c.detach().to(torch.float32).norm(dim=-1).sum()
                delta_norm_count += (B * H * (t1 - t0))

                m_state = m_new[:, :, :, -1]
                denom_state = denom_c[:, :, :, -1]
                numer_state = numer_c[:, :, :, -1, :]

            slotspace_delta_norm_mean = (delta_norm_sum / max(1, delta_norm_count)).detach().cpu()

        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info:
            info = {
                "write_logits_raw": write_logits_raw.detach(),
                "write_logits": write_logits.detach().to(torch.float32),
                "read_weights": read_weights.detach(),
                "slot_state_norm": slot_state_norm_t.detach().permute(0, 1, 3, 2).contiguous()
                if slot_state_norm_t is not None
                else None,
                "content_read_gamma": content_read_gamma.detach().to(torch.float32).cpu(),
                "slot_mask_where": slot_mask_where,
                "slot_mask_scope": slot_mask_scope,
            }
            if alibi_bias_applied is not None:
                info["alibi_bias_applied"] = alibi_bias_applied.detach().to(torch.float32)
            if self.use_alibi_write and self.learn_alibi_strength:
                info["alibi_strength"] = self._alibi_strength(
                    dtype=torch.float32, device=x.device
                ).detach().cpu()
            if self.use_slotspace_refine:
                info["slotspace_gate"] = self._slotspace_gate(
                    dtype=torch.float32, device=x.device
                ).detach().cpu()
                info["use_rope_slotspace"] = torch.tensor(bool(self.use_rope_slotspace))
                if slotspace_delta_norm_mean is not None:
                    info["slotspace_delta_norm"] = slotspace_delta_norm_mean

            info["read_logits"] = (
                read_logits_full.detach().to(torch.float32) if read_logits_full is not None else None
            )
            info["read_logits_key"] = (
                read_logits_key_full.detach().to(torch.float32) if read_logits_key_full is not None else None
            )
            info["read_logits_content"] = (
                read_logits_content_full.detach().to(torch.float32)
                if read_logits_content_full is not None
                else None
            )
            info["routing_mode"] = routing_mode

        return out, info


class AddressedStateAttentionOnline(nn.Module):
    """Training-efficient variant with online slotspace scan."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        write_chunk_size: int = 128,
        enable_compiled: bool = False,
        return_light_stats_default: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature)
        self.state_fp32 = bool(state_fp32)
        self.slot_dropout = float(slot_dropout)
        self.normalize_k = bool(normalize_k)

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.use_slotspace_refine = bool(use_slotspace_refine)
        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)
        self.use_rope_slotspace = bool(use_rope_slotspace) and self.use_slotspace_refine

        self.write_chunk_size = int(write_chunk_size)
        self.return_light_stats_default = bool(return_light_stats_default)

        H, K, d = self.num_heads, self.num_slots, self.head_dim

        self.slot_keys = nn.Parameter(torch.randn(H, K, d) / math.sqrt(d))
        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(d, base=rope_base) if self.use_rope_keys else None

        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(H), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(H), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        if self.use_slotspace_refine:
            self.slot_in = nn.Linear(K, self.slotspace_dim, bias=False)
            self.slot_q = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, K, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))
            if self.use_rope_slotspace:
                if self.slotspace_dim % 2 != 0:
                    raise ValueError("slotspace_dim must be even when using RoPE")
                self.rope_slotspace = RotaryEmbedding(self.slotspace_dim, base=float(rope_base_slotspace))
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = None
            self.slot_k = None
            self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

        self._compiled = None
        if enable_compiled:
            self.enable_compiled_kernel()

    def enable_compiled_kernel(self) -> None:
        if self._compiled is None:
            self._compiled = torch.compile(self._write_read_chunk, fullgraph=False)

    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(getattr(self, "alibi_strength", 0.0), dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    def _write_read_chunk(
        self,
        wlog_c: torch.Tensor,
        v_c: torch.Tensor,
        q_read_c: torch.Tensor,
        slot_keys: torch.Tensor,
        content_gamma: torch.Tensor,
        rtemp: float,
        m_state: torch.Tensor,
        denom_state: torch.Tensor,
        numer_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, K, L = wlog_c.shape
        d = numer_state.shape[-1]
        state_dtype = numer_state.dtype

        m_c, _ = torch.cummax(wlog_c, dim=-1)
        m_new = torch.maximum(m_state.unsqueeze(-1), m_c)
        scale = torch.exp(m_state.unsqueeze(-1) - m_new)

        denom_c = denom_state.unsqueeze(-1) * scale
        numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

        w_new = self._safe_exp_sub_max(wlog_c, m_new)
        denom_c = denom_c + torch.cumsum(w_new, dim=-1)

        add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
        numer_c = numer_c + add

        slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
        slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

        read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)
        if self.use_content_read:
            read_logits_content = torch.einsum(
                "bhld,bhlkd->bhlk", q_read_c, slot_state_t.to(q_read_c.dtype)
            ) / math.sqrt(d)
            read_logits = read_logits_key + content_gamma.to(read_logits_key.dtype) * read_logits_content
        else:
            read_logits = read_logits_key

        read_w_c = torch.softmax(read_logits / rtemp, dim=-1)
        out_base_c = torch.einsum("bhlk,bhlkd->bhld", read_w_c.to(state_dtype), slot_state_t)

        m_state_new = m_new[:, :, :, -1]
        denom_state_new = denom_c[:, :, :, -1]
        numer_state_new = numer_c[:, :, :, -1, :]

        return out_base_c, read_w_c, slot_state_t, m_state_new, denom_state_new, numer_state_new

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        return_light_stats: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if return_light_stats is None:
            return_light_stats = self.return_light_stats_default

        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)
        q_read = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = torch.rand((H, K), device=x.device) < self.slot_dropout
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype) / max(1e-6, self.write_temperature)

        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos = torch.arange(T, device=x.device, dtype=state_dtype)
            write_logits = write_logits + slopes.view(1, H, 1, 1) * pos.view(1, 1, 1, T)

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        content_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        if self.use_slotspace_refine:
            M = self.slotspace_dim
            slotspace_dtype = state_dtype
            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)
            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)
            if self.use_rope_slotspace:
                cos_s_full, sin_s_full = self.rope_slotspace.get_cos_sin(
                    T, device=x.device, dtype=slotspace_dtype
                )
            else:
                cos_s_full = sin_s_full = None

            if return_info and return_light_stats:
                delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
                delta_norm_count = 0

        if return_info and return_light_stats:
            entropy_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            top1_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            stat_count = 0

        WRITE_CHUNK = self.write_chunk_size
        kernel = self._compiled if self._compiled is not None else self._write_read_chunk

        for t0 in range(0, T, WRITE_CHUNK):
            t1 = min(T, t0 + WRITE_CHUNK)
            wlog_c = write_logits[:, :, :, t0:t1]
            v_c = v_write[:, :, t0:t1, :].to(state_dtype)
            q_c = q_read[:, :, t0:t1, :]

            out_base_c, rw_c, slot_state_t, m_state, denom_state, numer_state = kernel(
                wlog_c, v_c, q_c, slot_keys, content_gamma, rtemp, m_state, denom_state, numer_state
            )

            out_c = out_base_c

            if self.use_slotspace_refine:
                u_c = self.slot_in(rw_c.to(slotspace_dtype))
                q_s = self.slot_q(u_c)
                k_s = self.slot_k(u_c)
                v_s = self.slot_v(u_c)

                if self.use_rope_slotspace:
                    cos_s = cos_s_full[:, :, t0:t1, :]
                    sin_s = sin_s_full[:, :, t0:t1, :]
                    q_s = apply_rope(q_s, cos_s, sin_s)
                    k_s = apply_rope(k_s, cos_s, sin_s)

                if valid is not None:
                    mask_c = valid[:, t0:t1].view(B, 1, t1 - t0, 1).to(slotspace_dtype)
                    q_s = q_s * mask_c
                    k_s = k_s * mask_c
                    v_s = v_s * mask_c

                qf_c = phi(q_s)
                kf_c = phi(k_s)

                kv_c = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_s.to(slotspace_dtype))
                S_c = torch.cumsum(kv_c, dim=2) + S_state.unsqueeze(2)
                Z_c = torch.cumsum(kf_c, dim=2) + Z_state.unsqueeze(2)
                Z_c = Z_c.clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1)
                u2_c = num / den.clamp_min(1e-8)

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

                u2_c = self.slotspace_dropout(u2_c)

                slot_w_c = self.slot_out(u2_c)
                if self.slotspace_signed_weights:
                    slot_w_c = torch.tanh(slot_w_c)
                else:
                    slot_w_c = torch.softmax(slot_w_c, dim=-1)

                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c.to(state_dtype), slot_state_t)
                out_c = out_c + gate * delta_c

                if return_info and return_light_stats:
                    delta_norm_sum += delta_c.detach().to(torch.float32).norm(dim=-1).sum()
                    delta_norm_count += (B * H * (t1 - t0))

            out_h[:, :, t0:t1, :] = out_c

            if return_info and return_light_stats:
                p = rw_c.clamp_min(1e-8)
                ent = -(p * p.log()).sum(dim=-1).mean()
                top = rw_c.argmax(dim=-1).reshape(-1)
                hist = torch.bincount(top, minlength=K).float()
                top1 = hist.max() / hist.sum().clamp_min(1.0)
                entropy_sum += ent.detach().to(torch.float32)
                top1_sum += top1.detach().to(torch.float32)
                stat_count += 1

        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info:
            info = {}
            if return_light_stats:
                info["alibi_strength_mean"] = self._alibi_strength(
                    dtype=torch.float32, device=x.device
                ).detach().cpu()
                info["content_read_gamma_mean"] = self._content_read_gamma(
                    dtype=torch.float32, device=x.device
                ).detach().cpu()
                info["entropy_mean"] = (entropy_sum / max(1, stat_count)).detach().cpu()
                info["top1freq_mean"] = (top1_sum / max(1, stat_count)).detach().cpu()
                if self.use_slotspace_refine:
                    info["slotspace_gate_mean"] = self._slotspace_gate(
                        dtype=torch.float32, device=x.device
                    ).detach().cpu()
                    if "delta_norm_sum" in locals():
                        info["slotspace_delta_norm"] = (
                            delta_norm_sum / max(1, delta_norm_count)
                        ).detach().cpu()

        return out, info

class AddressedStateAttentionIntervene(nn.Module):
    """Refine-geometry logging + refine-delta intervention (orth/par gating)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature)
        self.state_fp32 = bool(state_fp32)
        self.slot_dropout = float(slot_dropout)
        self.normalize_k = bool(normalize_k)
        self.routing_override = None

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.use_slotspace_refine = bool(use_slotspace_refine)
        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)

        self.write_chunk_size = int(write_chunk_size)
        self.slotspace_chunk_size = int(slotspace_chunk_size)

        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) / math.sqrt(self.head_dim)
        )

        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope_keys else None

        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(num_heads), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(num_heads), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        self.use_rope_slotspace = bool(use_rope_slotspace) and bool(self.use_slotspace_refine)
        if self.use_slotspace_refine:
            self.slot_in = nn.Linear(num_slots, self.slotspace_dim, bias=False)
            self.slot_q = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, num_slots, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))

            if self.use_rope_slotspace:
                if self.slotspace_dim % 2 != 0:
                    raise ValueError("use_rope_slotspace requires even slotspace_dim")
                self.rope_slotspace = RotaryEmbedding(
                    self.slotspace_dim, base=float(rope_base_slotspace)
                )
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = None
            self.slot_k = None
            self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

        # ---------------------------
        # Intervention config defaults
        # ---------------------------
        self._intv_mode = "off"              # "off" | "delta_par" | "delta_orth" | "orth_gate" | ...
        self._intv_beta = 1.0                # orth multiplier (post-mask)
        self._intv_par_beta = 1.0            # parallel multiplier (always applied in orth_gate)
        self._intv_score_kind = "orth_frac"  # "orth_frac" | "orth_ratio" | "alpha_abs" | "slot_peaked"
        self._intv_tau_kind = "pctl"         # "abs" | "pctl"
        self._intv_tau = 0.15                # float | Tensor[H] | list/tuple per-layer
        self._intv_tau_pctl = 75.0           # percentile for tk="pctl"
        self._intv_tau_per_head = True       # percentile computed per head if tk="pctl"
        self._intv_mask_mode = "soft"        # "hard" | "soft"
        self._intv_soft_temp = 0.05
        self._intv_head_mask = None          # None | Tensor[H] in {0,1}
        self._intv_score_clip_pctl = 99.0    # clip score at percentile (scalar or per-head via per-head quantile)
        self._log_refine_geom = False        # if True, emit geom_* summaries in info

    # ----------------------------
    # Small helpers
    # ----------------------------
    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(self.alibi_strength, dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    def _default_info_cfg(self) -> Dict:
        return dict(
            store_read_weights=True,
            store_read_logits=True,
            store_write_logits=True,
            store_slot_state_norm=True,
            store_out1=False,
            store_delta=False,
            store_slot_w=False,
            store_intv_raw=False,       # if True, store raw score/mask/alpha trajectories
            detach_to_cpu=False,
            time_stride=1,
            batch_stride=1,
        )

    def _store_tensor(self, t: Optional[torch.Tensor], *, cfg: Dict, kind: str) -> Optional[torch.Tensor]:
        if t is None:
            return None
        bstride = int(cfg.get("batch_stride", 1))
        tstride = int(cfg.get("time_stride", 1))
        to_cpu = bool(cfg.get("detach_to_cpu", False))

        x = t
        if x.dim() >= 1 and bstride > 1:
            x = x[::bstride]
        if x.dim() == 4 and tstride > 1:
            if kind == "bhtk":
                x = x[:, :, ::tstride, :]
            elif kind == "bhkt":
                x = x[:, :, :, ::tstride]
        x = x.detach()
        if to_cpu:
            x = x.to("cpu", non_blocking=True)
        return x

    # ----------------------------
    # Tau resolution
    # ----------------------------
    def _resolve_tau_spec(self, layer_idx: Optional[int]):
        tau_spec = getattr(self, "_intv_tau", 0.15)
        if isinstance(tau_spec, (list, tuple)):
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when _intv_tau is a per-layer list/tuple")
            if layer_idx < 0 or layer_idx >= len(tau_spec):
                raise ValueError(f"layer_idx={layer_idx} out of range for _intv_tau (len={len(tau_spec)})")
            tau_spec = tau_spec[layer_idx]
        return tau_spec

    def _compute_tau_from_score(
        self,
        score: torch.Tensor,  # [B,H,L]
        *,
        tau_kind: str,
        tau_pctl: float,
        layer_idx: Optional[int],
    ) -> torch.Tensor:
        tau_spec = self._resolve_tau_spec(layer_idx)

        if tau_kind == "abs":
            if torch.is_tensor(tau_spec):
                return tau_spec.to(device=score.device, dtype=score.dtype)  # [H]
            return torch.tensor(float(tau_spec), device=score.device, dtype=score.dtype)  # scalar

        if tau_kind == "pctl":
            if torch.is_tensor(tau_spec):
                return tau_spec.to(device=score.device, dtype=score.dtype)  # [H]

            p = float(tau_pctl) / 100.0
            if getattr(self, "_intv_tau_per_head", True):
                flat = score.detach().reshape(score.shape[0], score.shape[1], -1)  # [B,H,BL]
                tau_bh = torch.quantile(flat, p, dim=-1)  # [B,H]
                tau_h = tau_bh.mean(dim=0)                # [H]
                return tau_h.to(device=score.device, dtype=score.dtype)
            else:
                return torch.quantile(score.detach().flatten(), p).to(device=score.device, dtype=score.dtype)

        raise ValueError(f"Unknown _intv_tau_kind={tau_kind}")

    def _clip_score(self, score: torch.Tensor) -> torch.Tensor:
        clip_p = getattr(self, "_intv_score_clip_pctl", None)
        if clip_p is None:
            return score
        clip_p = float(clip_p)
        if not (0.0 < clip_p < 100.0):
            return score

        p = clip_p / 100.0
        if score.ndim != 3:
            return score

        if getattr(self, "_intv_tau_per_head", True):
            flat = score.detach().reshape(score.shape[0], score.shape[1], -1)  # [B,H,BL]
            smax_bh = torch.quantile(flat, p, dim=-1)                          # [B,H]
            smax_h = smax_bh.mean(dim=0).to(dtype=score.dtype)                 # [H]
            smax = smax_h.view(1, -1, 1)
            return torch.minimum(score, smax)
        else:
            smax = torch.quantile(score.detach().flatten(), p).to(dtype=score.dtype)
            return torch.clamp(score, max=smax)

    # ----------------------------
    # Intervention core
    # ----------------------------
    def _apply_refine_intervention(
        self,
        out1: torch.Tensor,                     # [B,H,L,d]
        delta: torch.Tensor,                    # [B,H,L,d]
        slot_w_logits: Optional[torch.Tensor],  # [B,H,L,K] (pre-softmax logits), optional
        *,
        layer_idx: Optional[int] = None,
        store_raw: bool = False,
    ):
        eps = 1e-8
        B, H, L, _ = out1.shape

        hm = getattr(self, "_intv_head_mask", None)
        if hm is not None:
            hm = hm.to(device=out1.device).view(1, H, 1, 1).to(dtype=out1.dtype)

        out1_norm2 = (out1 * out1).sum(dim=-1, keepdim=True).clamp_min(eps)
        alpha = (delta * out1).sum(dim=-1, keepdim=True) / out1_norm2
        delta_par = alpha * out1
        delta_orth = delta - delta_par

        logs = {}

        if getattr(self, "_log_refine_geom", False):
            out1n = out1.norm(dim=-1).clamp_min(eps)                # [B,H,L]
            dn = delta.norm(dim=-1).clamp_min(eps)                  # [B,H,L]
            dparn = delta_par.norm(dim=-1)                          # [B,H,L]
            dorthn = delta_orth.norm(dim=-1)                        # [B,H,L]
            a = alpha.squeeze(-1)                                   # [B,H,L]
            logs.update(
                dict(
                    geom_alpha_mean=a.mean(dim=(0, 2)),             # [H]
                    geom_alpha_abs=a.abs().mean(dim=(0, 2)),        # [H]
                    geom_sign_pos=(a > 0).float().mean(dim=(0, 2)), # [H]
                    geom_orth_frac=(dorthn / dn).mean(dim=(0, 2)),  # [H]
                    geom_d_ratio=(dn / out1n).mean(dim=(0, 2)),     # [H]
                    geom_dpar_ratio=(dparn / dn).mean(dim=(0, 2)),  # [H]
                )
            )

        mode = getattr(self, "_intv_mode", "off")
        if mode is None or mode == "off":
            return delta, logs

        if mode == "delta_par":
            logs["alpha"] = alpha.squeeze(-1)
            delta_mod = delta_par

        elif mode == "delta_orth":
            logs["alpha"] = alpha.squeeze(-1)
            delta_mod = delta_orth

        elif mode == "delta_par_plus_orth":
            logs["alpha"] = alpha.squeeze(-1)
            delta_mod = delta_par + delta_orth

        elif mode == "orth_gate":
            beta = float(getattr(self, "_intv_beta", 1.0))
            par_beta = float(getattr(self, "_intv_par_beta", 1.0))
            sk = getattr(self, "_intv_score_kind", "orth_frac")

            out1n = out1.norm(dim=-1).clamp_min(eps)     # [B,H,L]
            dorthn = delta_orth.norm(dim=-1)             # [B,H,L]
            dn = delta.norm(dim=-1).clamp_min(eps)       # [B,H,L]

            if sk == "orth_ratio":
                score = dorthn / out1n
            elif sk == "orth_frac":
                score = dorthn / dn
            elif sk == "alpha_abs":
                score = alpha.abs().squeeze(-1)
            elif sk == "slot_peaked":
                if slot_w_logits is None:
                    raise ValueError("score_kind='slot_peaked' requires slot_w_logits (enable info_cfg.store_slot_w=True)")
                p = torch.softmax(slot_w_logits.float(), dim=-1).clamp_min(1e-8)  # [B,H,L,K]
                h_rw = -(p * p.log()).sum(dim=-1)                                 # [B,H,L]
                K = p.shape[-1]
                score = 1.0 - (h_rw / max(1e-8, math.log(K)))
                score = score.to(dtype=out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_score_kind={sk}")

            score = score.to(dtype=out1.dtype)
            score = self._clip_score(score)

            tk = getattr(self, "_intv_tau_kind", "pctl")
            tau_pctl = float(getattr(self, "_intv_tau_pctl", 75.0))
            tau = self._compute_tau_from_score(score, tau_kind=tk, tau_pctl=tau_pctl, layer_idx=layer_idx)

            if tau.ndim == 0:
                tau_b = tau
            elif tau.ndim == 1:
                if tau.numel() != H:
                    raise ValueError(f"per-head tau must be shape [H]={H}, got {tuple(tau.shape)}")
                tau_b = tau.view(1, H, 1)
            else:
                raise ValueError(f"tau must be scalar or [H], got {tuple(tau.shape)}")

            mm = getattr(self, "_intv_mask_mode", "soft")
            soft_temp = float(getattr(self, "_intv_soft_temp", 0.05))

            if mm == "hard":
                mask = (score > tau_b).to(out1.dtype)               # [B,H,L]
            elif mm == "soft":
                temp = max(1e-6, soft_temp)
                mask = torch.sigmoid((score - tau_b) / temp).to(out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_mask_mode={mm}")

            delta_mod = par_beta * delta_par + beta * mask.unsqueeze(-1) * delta_orth

            # lightweight always
            logs.update(dict(tau=tau.detach()))

            # raw only if requested
            if store_raw:
                logs.update(
                    dict(
                        score=score,
                        mask=mask,
                        alpha=alpha.squeeze(-1),
                        out1_norm=out1n,
                        dpar_norm=delta_par.norm(dim=-1),
                        dorth_norm=dorthn,
                    )
                )
        else:
            raise ValueError(f"Unknown _intv_mode={mode}")

        if hm is not None:
            delta_mod = hm * delta_mod + (1.0 - hm) * delta
            logs["head_mask"] = hm.squeeze(0).squeeze(-1).squeeze(-1).detach()

        return delta_mod, logs

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        info_level: str = "full",
        info_cfg: Optional[Dict] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        if info_cfg is None:
            info_cfg = self._default_info_cfg()

        store_read_weights = bool(info_cfg.get("store_read_weights", True))
        store_read_logits = bool(info_cfg.get("store_read_logits", True)) and (info_level in ("logits", "full"))
        store_write_logits = bool(info_cfg.get("store_write_logits", True)) and (info_level == "full")
        store_slot_norm = bool(info_cfg.get("store_slot_state_norm", True)) and (info_level == "full")
        store_out1 = bool(info_cfg.get("store_out1", False)) and return_info
        store_delta = bool(info_cfg.get("store_delta", False)) and return_info
        store_slot_w = bool(info_cfg.get("store_slot_w", False)) and return_info
        store_intv_raw = bool(info_cfg.get("store_intv_raw", False)) and return_info

        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)
        q_read = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = torch.rand((H, K), device=x.device) < self.slot_dropout
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype) / max(1e-6, self.write_temperature)

        alibi_bias_applied = None
        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos_i = torch.arange(T, device=x.device, dtype=state_dtype)
            alibi_bias = slopes.view(1, H, 1, 1) * pos_i.view(1, 1, 1, T)
            write_logits = write_logits + alibi_bias
            alibi_bias_applied = alibi_bias

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        content_read_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)

        out1_full = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_out1 else None
        delta_full = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_delta else None
        slot_w_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if store_slot_w else None

        need_read_weights = bool(self.use_slotspace_refine) or (return_info and store_read_weights)
        read_weights = torch.empty((B, H, T, K), device=x.device, dtype=q_read.dtype) if need_read_weights else None

        slot_state_norm_t = (
            torch.empty((B, H, T, K), device=x.device, dtype=torch.float32)
            if (return_info and store_slot_norm)
            else None
        )

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        if return_info and store_read_logits:
            read_logits_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_key_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_content_full = (
                torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if self.use_content_read else None
            )
        else:
            read_logits_full = None
            read_logits_key_full = None
            read_logits_content_full = None

        WRITE_CHUNK = self.write_chunk_size
        for t0 in range(0, T, WRITE_CHUNK):
            t1 = min(T, t0 + WRITE_CHUNK)
            wlog_c = write_logits[:, :, :, t0:t1]

            m_c, _ = torch.cummax(wlog_c, dim=-1)
            m_new = torch.maximum(m_state.unsqueeze(-1), m_c)
            scale = torch.exp(m_state.unsqueeze(-1) - m_new)

            denom_c = denom_state.unsqueeze(-1) * scale
            numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

            w_new = self._safe_exp_sub_max(wlog_c, m_new)
            denom_c = denom_c + torch.cumsum(w_new, dim=-1)

            v_c = v_write[:, :, t0:t1, :].to(state_dtype)
            add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
            numer_c = numer_c + add

            slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
            slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

            q_read_c = q_read[:, :, t0:t1, :]
            read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)

            read_logits_content = None
            read_logits = read_logits_key
            if self.use_content_read:
                read_logits_content = torch.einsum(
                    "bhld,bhlkd->bhlk", q_read_c, slot_state_t.to(q_read_c.dtype)
                ) / math.sqrt(d)
                read_logits = read_logits + content_read_gamma.to(read_logits.dtype) * read_logits_content

            if return_info and store_read_logits:
                read_logits_full[:, :, t0:t1, :] = read_logits.to(state_dtype)
                read_logits_key_full[:, :, t0:t1, :] = read_logits_key.to(state_dtype)
                if self.use_content_read:
                    read_logits_content_full[:, :, t0:t1, :] = read_logits_content.to(state_dtype)

            if routing_noise is not None:
                if routing_noise == "gumbel":
                    u = torch.rand_like(read_logits)
                    g = -torch.log(-torch.log(u.clamp_min(1e-8)).clamp_min(1e-8))
                    read_logits = read_logits + routing_noise_scale * g
                elif routing_noise == "gaussian":
                    read_logits = read_logits + routing_noise_scale * torch.randn_like(read_logits)
                else:
                    raise ValueError(f"Unknown routing_noise={routing_noise}")

            if self.routing_override is not None:
                if callable(self.routing_override):
                    ctx = {
                        "t0": t0,
                        "t1": t1,
                        "B": B,
                        "H": H,
                        "T": T,
                        "K": K,
                        "d": d,
                        "rtemp": rtemp,
                        "state_dtype": state_dtype,
                        "q_read_c": q_read_c,
                        "slot_keys": slot_keys,
                        "slot_state_t": slot_state_t,
                        "valid": valid,
                        "layer_idx": layer_idx,
                    }
                    read_w_c = self.routing_override(
                        t0, t1, read_logits, read_logits_key, read_logits_content, ctx
                    )
                else:
                    read_w_c = self.routing_override[:, :, t0:t1, :].to(read_logits.dtype)

                read_w_c = torch.nan_to_num(read_w_c, nan=0.0, posinf=0.0, neginf=0.0)
                read_w_c = read_w_c.clamp_min(0.0)
                read_w_c = read_w_c / read_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                if routing_mode == "softmax":
                    read_w_c = torch.softmax(read_logits / rtemp, dim=-1)
                elif routing_mode == "top1":
                    top = read_logits.argmax(dim=-1)
                    read_w_c = F.one_hot(top, num_classes=K).to(read_logits.dtype)
                elif routing_mode == "topk":
                    kk = max(1, min(K, int(routing_topk)))
                    vals, idx = torch.topk(read_logits, k=kk, dim=-1)
                    masked = torch.full_like(read_logits, float("-inf"))
                    masked.scatter_(-1, idx, vals)
                    read_w_c = torch.softmax(masked / rtemp, dim=-1)
                elif routing_mode == "external":
                    if read_weights_override is None:
                        raise ValueError("routing_mode='external' requires read_weights_override")
                    if read_weights_override.shape[-2] == T:
                        read_w_c = read_weights_override[:, :, t0:t1, :]
                    else:
                        read_w_c = read_weights_override
                    read_w_c = read_w_c / read_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    raise ValueError(f"Unknown routing_mode={routing_mode}")

            if read_weights is not None:
                read_weights[:, :, t0:t1, :] = read_w_c

            out_h[:, :, t0:t1, :] = torch.einsum(
                "bhlk,bhlkd->bhld", read_w_c.to(state_dtype), slot_state_t.to(state_dtype)
            )

            if out1_full is not None:
                out1_full[:, :, t0:t1, :] = out_h[:, :, t0:t1, :]

            if slot_state_norm_t is not None:
                slot_state_norm_t[:, :, t0:t1, :] = slot_state_t.to(torch.float32).norm(dim=-1)

            m_state = m_new[:, :, :, -1]
            denom_state = denom_c[:, :, :, -1]
            numer_state = numer_c[:, :, :, -1, :]

        # ----------------------------
        # Slotspace refine + intervention
        # ----------------------------
        slotspace_delta_norm_mean = None
        intv_logs_acc: Dict[str, torch.Tensor] = {}
        intv_logs_count = 0

        # RAW buffers live here when requested (this is the actual fix)
        intv_raw = None
        if store_intv_raw:
            intv_raw = {}

        if self.use_slotspace_refine:
            slotspace_dtype = state_dtype
            M = self.slotspace_dim
            if read_weights is None:
                raise RuntimeError("internal: read_weights required for slotspace refine")

            u = self.slot_in(read_weights.to(slotspace_dtype))
            q_s = self.slot_q(u)
            k_s = self.slot_k(u)
            v_s = self.slot_v(u)

            if self.use_rope_slotspace:
                cos_s, sin_s = self.rope_slotspace.get_cos_sin(T, device=x.device, dtype=q_s.dtype)
                q_s = apply_rope(q_s, cos_s, sin_s)
                k_s = apply_rope(k_s, cos_s, sin_s)

            qf = phi(q_s)
            kf = phi(k_s)

            if valid is not None:
                mask = valid.view(B, 1, T, 1).to(slotspace_dtype)
                qf = qf * mask
                kf = kf * mask
                v_s = v_s * mask

            u2 = torch.empty((B, H, T, M), device=x.device, dtype=slotspace_dtype)
            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)
            SS_CHUNK = self.slotspace_chunk_size

            for t0 in range(0, T, SS_CHUNK):
                t1 = min(T, t0 + SS_CHUNK)
                qf_c = qf[:, :, t0:t1, :]
                kf_c = kf[:, :, t0:t1, :]
                v_c = v_s[:, :, t0:t1, :]

                kv = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_c)
                S_c = torch.cumsum(kv, dim=2) + S_state.unsqueeze(2)
                Z_c = (torch.cumsum(kf_c, dim=2) + Z_state.unsqueeze(2)).clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1).clamp_min(1e-8)
                u2[:, :, t0:t1, :] = num / den

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

            u2 = self.slotspace_dropout(u2)
            slot_w_logits = self.slot_out(u2)  # [B,H,T,K]

            if slot_w_full is not None:
                slot_w_full[:] = slot_w_logits.to(state_dtype)

            if self.slotspace_signed_weights:
                slot_w_eff = torch.tanh(slot_w_logits)
            else:
                slot_w_eff = torch.softmax(slot_w_logits, dim=-1)

            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)

            # Recompute slot_state scan for refine delta
            denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
            numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
            m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

            delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            delta_norm_count = 0

            for t0 in range(0, T, WRITE_CHUNK):
                t1 = min(T, t0 + WRITE_CHUNK)
                wlog_c = write_logits[:, :, :, t0:t1]

                m_c, _ = torch.cummax(wlog_c, dim=-1)
                m_new = torch.maximum(m_state.unsqueeze(-1), m_c)

                scale = torch.exp(m_state.unsqueeze(-1) - m_new)
                denom_c = denom_state.unsqueeze(-1) * scale
                numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

                w_new = self._safe_exp_sub_max(wlog_c, m_new)
                denom_c = denom_c + torch.cumsum(w_new, dim=-1)

                v_c = v_write[:, :, t0:t1, :].to(state_dtype)
                add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
                numer_c = numer_c + add

                slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
                slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

                slot_w_c = slot_w_eff[:, :, t0:t1, :].to(state_dtype)
                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c, slot_state_t.to(state_dtype))
                delta = gate * delta_c

                if delta_full is not None:
                    delta_full[:, :, t0:t1, :] = delta

                # Apply intervention
                delta_mod, logs = self._apply_refine_intervention(
                    out1=out_h[:, :, t0:t1, :],
                    delta=delta,
                    slot_w_logits=slot_w_logits[:, :, t0:t1, :] if (store_slot_w or store_intv_raw) else None,
                    layer_idx=layer_idx,
                    store_raw=store_intv_raw,
                )
                out_h[:, :, t0:t1, :] = out_h[:, :, t0:t1, :] + delta_mod

                # -----------------------------
                # FIX: store raw AND still accumulate summaries
                # -----------------------------
                if return_info and logs:
                    for klog, v in logs.items():
                        if not torch.is_tensor(v):
                            continue

                        if v.ndim == 3:
                            # raw per-token trajectories [B,H,Lchunk] -> buffer [B,H,T]
                            if store_intv_raw:
                                if klog not in intv_raw:
                                    intv_raw[klog] = torch.empty((v.shape[0], v.shape[1], T), device=v.device, dtype=v.dtype)
                                intv_raw[klog][:, :, t0:t1] = v.detach()

                            # always accumulate per-head summary [H]
                            vv = v.detach().to(torch.float32).mean(dim=(0, 2))  # [H]

                        elif v.ndim == 1:
                            vv = v.detach().to(torch.float32)  # [H]

                        elif v.ndim == 0:
                            vv = v.detach().to(torch.float32)  # scalar

                        else:
                            continue

                        if klog not in intv_logs_acc:
                            intv_logs_acc[klog] = vv
                        else:
                            intv_logs_acc[klog] = intv_logs_acc[klog] + vv

                    intv_logs_count += 1

                delta_norm_sum = delta_norm_sum + delta.detach().to(torch.float32).norm(dim=-1).sum()
                delta_norm_count += (B * H * (t1 - t0))

                m_state = m_new[:, :, :, -1]
                denom_state = denom_c[:, :, :, -1]
                numer_state = numer_c[:, :, :, -1, :]

            slotspace_delta_norm_mean = (delta_norm_sum / max(1, delta_norm_count)).detach().cpu()

        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info:
            info = {
                "read_weights": None,
                "content_read_gamma": content_read_gamma.detach().to(torch.float32).cpu(),
                "routing_mode": routing_mode,
                "intv_mode": getattr(self, "_intv_mode", "off"),
            }

            if alibi_bias_applied is not None and info_level == "full":
                info["alibi_bias_applied"] = self._store_tensor(
                    alibi_bias_applied.to(torch.float32), cfg=info_cfg, kind="other"
                )

            if self.use_alibi_write and self.learn_alibi_strength:
                info["alibi_strength"] = self._alibi_strength(dtype=torch.float32, device=x.device).detach().cpu()

            if self.use_slotspace_refine:
                info["slotspace_gate"] = self._slotspace_gate(dtype=torch.float32, device=x.device).detach().cpu()
                info["use_rope_slotspace"] = torch.tensor(bool(self.use_rope_slotspace))
                if slotspace_delta_norm_mean is not None:
                    info["slotspace_delta_norm"] = slotspace_delta_norm_mean

            if store_read_weights and (read_weights is not None):
                info["read_weights"] = self._store_tensor(read_weights, cfg=info_cfg, kind="bhtk")

            if store_slot_norm and (slot_state_norm_t is not None):
                s = slot_state_norm_t.permute(0, 1, 3, 2).contiguous()
                info["slot_state_norm"] = self._store_tensor(s, cfg=info_cfg, kind="bhkt")
            else:
                info["slot_state_norm"] = None

            if store_read_logits and (read_logits_full is not None):
                info["read_logits"] = self._store_tensor(read_logits_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                info["read_logits_key"] = self._store_tensor(
                    read_logits_key_full.to(torch.float32), cfg=info_cfg, kind="bhtk"
                )
                info["read_logits_content"] = (
                    self._store_tensor(read_logits_content_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                    if read_logits_content_full is not None
                    else None
                )
            else:
                info["read_logits"] = None
                info["read_logits_key"] = None
                info["read_logits_content"] = None

            if store_write_logits and (info_level == "full"):
                info["write_logits_raw"] = self._store_tensor(write_logits_raw, cfg=info_cfg, kind="bhkt")
                info["write_logits"] = self._store_tensor(write_logits.to(torch.float32), cfg=info_cfg, kind="bhkt")
            else:
                info["write_logits_raw"] = None
                info["write_logits"] = None

            if out1_full is not None:
                info["out1"] = self._store_tensor(out1_full.to(torch.float32), cfg=info_cfg, kind="other")
            else:
                info["out1"] = None

            if delta_full is not None:
                info["delta"] = self._store_tensor(delta_full.to(torch.float32), cfg=info_cfg, kind="other")
            else:
                info["delta"] = None

            if slot_w_full is not None:
                info["slot_w"] = self._store_tensor(slot_w_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
            else:
                info["slot_w"] = None

            info["layer_idx"] = torch.tensor(-1 if layer_idx is None else int(layer_idx))

            # averaged summaries
            if intv_logs_count > 0:
                for klog, v in intv_logs_acc.items():
                    info[klog] = (v / float(intv_logs_count)).detach().cpu()

            # -----------------------------
            # FIX: export raw tensors when requested
            # -----------------------------
            if store_intv_raw:
                info["intv_raw_enabled"] = torch.tensor(True)
                if intv_raw is not None:
                    name_map = {
                        "score": "intv_score",
                        "mask": "intv_mask",
                        "alpha": "intv_alpha",
                        "out1_norm": "intv_out1_norm",
                        "dpar_norm": "intv_dpar_norm",
                        "dorth_norm": "intv_dorth_norm",
                    }
                    for kraw, buf in intv_raw.items():
                        info[name_map.get(kraw, kraw)] = buf.detach().to("cpu", non_blocking=True)
            else:
                info["intv_raw_enabled"] = torch.tensor(False)

        return out, info
        

class AddressedStateAttentionOldIntervene(nn.Module):
    """Refine-geometry logging + refine-delta intervention (orth/par gating)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature)
        self.state_fp32 = bool(state_fp32)
        self.slot_dropout = float(slot_dropout)
        self.normalize_k = bool(normalize_k)
        self.routing_override = None

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.use_slotspace_refine = bool(use_slotspace_refine)
        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)

        self.write_chunk_size = int(write_chunk_size)
        self.slotspace_chunk_size = int(slotspace_chunk_size)

        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) / math.sqrt(self.head_dim)
        )

        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope_keys else None

        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(num_heads), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(num_heads), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        self.use_rope_slotspace = bool(use_rope_slotspace) and bool(self.use_slotspace_refine)
        if self.use_slotspace_refine:
            self.slot_in = nn.Linear(num_slots, self.slotspace_dim, bias=False)
            self.slot_q = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, num_slots, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))

            if self.use_rope_slotspace:
                if self.slotspace_dim % 2 != 0:
                    raise ValueError("use_rope_slotspace requires even slotspace_dim")
                self.rope_slotspace = RotaryEmbedding(
                    self.slotspace_dim, base=float(rope_base_slotspace)
                )
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = None
            self.slot_k = None
            self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

        self._intv_mode = "off"
        self._intv_beta = 1.0
        self._intv_score_kind = "orth_frac"
        self._intv_tau_kind = "pctl"
        self._intv_tau = 0.15
        self._intv_tau_pctl = 75.0
        self._intv_mask_mode = "soft"
        self._intv_soft_temp = 0.05
        self._intv_par_beta = 1.0
        self._intv_head_mask = None
        self._intv_score_clip_pctl = 99.0
        self._log_refine_geom = False

    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(self.alibi_strength, dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    def _default_info_cfg(self) -> Dict:
        return dict(
            store_read_weights=True,
            store_read_logits=True,
            store_write_logits=True,
            store_slot_state_norm=True,
            store_out1=False,
            store_delta=False,
            store_slot_w=False,
            detach_to_cpu=False,
            time_stride=1,
            batch_stride=1,
        )

    def _store_tensor(self, t: Optional[torch.Tensor], *, cfg: Dict, kind: str) -> Optional[torch.Tensor]:
        if t is None:
            return None
        bstride = int(cfg.get("batch_stride", 1))
        tstride = int(cfg.get("time_stride", 1))
        to_cpu = bool(cfg.get("detach_to_cpu", False))

        x = t
        if x.dim() >= 1 and bstride > 1:
            x = x[::bstride]
        if x.dim() == 4 and tstride > 1:
            if kind == "bhtk":
                x = x[:, :, ::tstride, :]
            elif kind == "bhkt":
                x = x[:, :, :, ::tstride]
        x = x.detach()
        if to_cpu:
            x = x.to("cpu", non_blocking=True)
        return x

    def _apply_refine_intervention(
        self,
        out1: torch.Tensor,
        delta: torch.Tensor,
        slot_w: Optional[torch.Tensor],
    ):
        eps = 1e-8
        B, H, L, _ = out1.shape

        hm = getattr(self, "_intv_head_mask", None)
        if hm is not None:
            hm = hm.to(device=out1.device)
            hm = hm.view(1, H, 1, 1).to(dtype=out1.dtype)
        else:
            hm = None

        out1_norm2 = (out1 * out1).sum(dim=-1, keepdim=True).clamp_min(eps)
        alpha = (delta * out1).sum(dim=-1, keepdim=True) / out1_norm2
        delta_par = alpha * out1
        delta_orth = delta - delta_par

        logs = None
        if getattr(self, "_log_refine_geom", False):
            out1n = out1.norm(dim=-1).clamp_min(eps)
            dn = delta.norm(dim=-1).clamp_min(eps)
            dparn = delta_par.norm(dim=-1)
            dorthn = delta_orth.norm(dim=-1)
            a = alpha.squeeze(-1)
            logs = {} if logs is None else logs
            logs.update(
                dict(
                    geom_alpha_mean=a.mean(dim=(0, 2)),
                    geom_alpha_abs=a.abs().mean(dim=(0, 2)),
                    geom_sign_pos=(a > 0).float().mean(dim=(0, 2)),
                    geom_orth_frac=(dorthn / dn).mean(dim=(0, 2)),
                    geom_d_ratio=(dn / out1n).mean(dim=(0, 2)),
                    geom_dpar_ratio=(dparn / dn).mean(dim=(0, 2)),
                )
            )

        mode = getattr(self, "_intv_mode", "off")
        if mode is None or mode == "off":
            return delta, logs

        if mode == "delta_par":
            delta_mod = delta_par
            logs = {} if logs is None else logs
            logs.update(dict(alpha=alpha.squeeze(-1)))

        elif mode == "delta_orth":
            delta_mod = delta_orth
            logs = {} if logs is None else logs
            logs.update(dict(alpha=alpha.squeeze(-1)))

        elif mode == "delta_par_plus_orth":
            delta_mod = delta_par + delta_orth
            logs = {} if logs is None else logs
            logs.update(dict(alpha=alpha.squeeze(-1)))

        elif mode == "orth_gate":
            beta = float(getattr(self, "_intv_beta", 1.0))
            sk = getattr(self, "_intv_score_kind", "orth_frac")
            out1n = out1.norm(dim=-1).clamp_min(eps)
            dorthn = delta_orth.norm(dim=-1)
            dn = delta.norm(dim=-1).clamp_min(eps)

            if sk == "orth_ratio":
                score = dorthn / out1n
            elif sk == "orth_frac":
                score = dorthn / dn
            elif sk == "alpha_abs":
                score = alpha.abs().squeeze(-1)
            elif sk == "slot_peaked":
                if slot_w is None:
                    raise ValueError("score_kind='slot_peaked' requires slot_w (enable info_cfg.store_slot_w=True)")
                p = torch.softmax(slot_w.float(), dim=-1).clamp_min(1e-8)
                h_rw = -(p * p.log()).sum(dim=-1)
                K = p.shape[-1]
                score = 1.0 - (h_rw / max(1e-8, math.log(K)))
                score = score.to(dtype=out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_score_kind={sk}")

            clip_p = getattr(self, "_intv_score_clip_pctl", None)
            if clip_p is not None:
                clip_p = float(clip_p)
                if 0.0 < clip_p < 100.0:
                    smax = torch.quantile(score.detach().flatten(), clip_p / 100.0).to(score.dtype)
                    score = torch.clamp(score, max=smax)

            tk = getattr(self, "_intv_tau_kind", "pctl")
            tau_abs = float(getattr(self, "_intv_tau", 0.15))
            tau_pctl = float(getattr(self, "_intv_tau_pctl", 75.0))

            if tk == "abs":
                tau = torch.tensor(tau_abs, device=score.device, dtype=score.dtype)
            elif tk == "pctl":
                tau = torch.quantile(score.detach().flatten(), tau_pctl / 100.0).to(score.dtype)
            else:
                raise ValueError(f"Unknown _intv_tau_kind={tk}")

            mm = getattr(self, "_intv_mask_mode", "soft")
            soft_temp = float(getattr(self, "_intv_soft_temp", 0.05))
            if mm == "hard":
                mask = (score > tau).to(out1.dtype)
            elif mm == "soft":
                temp = max(1e-6, soft_temp)
                mask = torch.sigmoid((score - tau) / temp).to(out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_mask_mode={mm}")

            par_beta = float(getattr(self, "_intv_par_beta", 1.0))
            delta_mod = par_beta * delta_par + beta * mask.unsqueeze(-1) * delta_orth

            logs = {} if logs is None else logs
            logs.update(
                dict(
                    score=score,
                    tau=tau,
                    mask=mask,
                    alpha=alpha.squeeze(-1),
                    out1_norm=out1n,
                    dpar_norm=delta_par.norm(dim=-1),
                    dorth_norm=dorthn,
                )
            )
        else:
            raise ValueError(f"Unknown _intv_mode={mode}")

        if hm is not None:
            delta_mod = hm * delta_mod + (1.0 - hm) * delta
            logs = {} if logs is None else logs
            logs["head_mask"] = hm.squeeze(0).squeeze(-1).squeeze(-1).detach()

        return delta_mod, logs

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        info_level: str = "full",
        info_cfg: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        if info_cfg is None:
            info_cfg = self._default_info_cfg()

        store_read_weights = bool(info_cfg.get("store_read_weights", True))
        store_read_logits = bool(info_cfg.get("store_read_logits", True)) and (info_level in ("logits", "full"))
        store_write_logits = bool(info_cfg.get("store_write_logits", True)) and (info_level == "full")
        store_slot_norm = bool(info_cfg.get("store_slot_state_norm", True)) and (info_level == "full")

        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)
        q_read = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = torch.rand((H, K), device=x.device) < self.slot_dropout
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype) / max(1e-6, self.write_temperature)

        alibi_bias_applied = None
        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos_i = torch.arange(T, device=x.device, dtype=state_dtype)
            alibi_bias = slopes.view(1, H, 1, 1) * pos_i.view(1, 1, 1, T)
            write_logits = write_logits + alibi_bias
            alibi_bias_applied = alibi_bias

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        content_read_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)
        store_out1 = bool(info_cfg.get("store_out1", False)) and return_info
        store_delta = bool(info_cfg.get("store_delta", False)) and return_info
        store_slot_w = bool(info_cfg.get("store_slot_w", False)) and return_info

        out1_full = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_out1 else None
        delta_full = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_delta else None
        slot_w_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if store_slot_w else None

        need_read_weights_for_compute = bool(self.use_slotspace_refine)
        need_read_weights_for_info = bool(return_info and store_read_weights)
        need_read_weights = need_read_weights_for_compute or need_read_weights_for_info
        read_weights = torch.empty((B, H, T, K), device=x.device, dtype=q_read.dtype) if need_read_weights else None

        slot_state_norm_t = (
            torch.empty((B, H, T, K), device=x.device, dtype=torch.float32)
            if (return_info and store_slot_norm)
            else None
        )

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        if return_info and store_read_logits:
            read_logits_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_key_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_content_full = (
                torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if self.use_content_read else None
            )
        else:
            read_logits_full = None
            read_logits_key_full = None
            read_logits_content_full = None

        WRITE_CHUNK = self.write_chunk_size
        for t0 in range(0, T, WRITE_CHUNK):
            t1 = min(T, t0 + WRITE_CHUNK)
            wlog_c = write_logits[:, :, :, t0:t1]

            m_c, _ = torch.cummax(wlog_c, dim=-1)
            m_new = torch.maximum(m_state.unsqueeze(-1), m_c)
            scale = torch.exp(m_state.unsqueeze(-1) - m_new)

            denom_c = denom_state.unsqueeze(-1) * scale
            numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

            w_new = self._safe_exp_sub_max(wlog_c, m_new)
            denom_c = denom_c + torch.cumsum(w_new, dim=-1)
            v_c = v_write[:, :, t0:t1, :].to(state_dtype)
            add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
            numer_c = numer_c + add

            slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
            slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

            q_read_c = q_read[:, :, t0:t1, :]
            read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)
            read_logits_content = None
            read_logits = read_logits_key
            if self.use_content_read:
                read_logits_content = torch.einsum(
                    "bhld,bhlkd->bhlk", q_read_c, slot_state_t.to(q_read_c.dtype)
                ) / math.sqrt(d)
                read_logits = read_logits + content_read_gamma.to(read_logits.dtype) * read_logits_content

            if return_info and store_read_logits:
                read_logits_full[:, :, t0:t1, :] = read_logits.to(state_dtype)
                read_logits_key_full[:, :, t0:t1, :] = read_logits_key.to(state_dtype)
                if self.use_content_read:
                    read_logits_content_full[:, :, t0:t1, :] = read_logits_content.to(state_dtype)

            if routing_noise is not None:
                if routing_noise == "gumbel":
                    u = torch.rand_like(read_logits)
                    g = -torch.log(-torch.log(u.clamp_min(1e-8)).clamp_min(1e-8))
                    read_logits = read_logits + routing_noise_scale * g
                elif routing_noise == "gaussian":
                    read_logits = read_logits + routing_noise_scale * torch.randn_like(read_logits)
                else:
                    raise ValueError(f"Unknown routing_noise={routing_noise}")

            if self.routing_override is not None:
                if callable(self.routing_override):
                    ctx = {
                        "t0": t0,
                        "t1": t1,
                        "B": B,
                        "H": H,
                        "T": T,
                        "K": K,
                        "d": d,
                        "rtemp": rtemp,
                        "state_dtype": state_dtype,
                        "q_read_c": q_read_c,
                        "slot_keys": slot_keys,
                        "slot_state_t": slot_state_t,
                        "valid": valid,
                    }
                    read_w_c = self.routing_override(
                        t0, t1, read_logits, read_logits_key, read_logits_content, ctx
                    )
                else:
                    read_w_c = self.routing_override[:, :, t0:t1, :].to(read_logits.dtype)

                read_w_c = torch.nan_to_num(read_w_c, nan=0.0, posinf=0.0, neginf=0.0)
                read_w_c = read_w_c.clamp_min(0.0)
                read_w_c = read_w_c / read_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                if routing_mode == "softmax":
                    read_w_c = torch.softmax(read_logits / rtemp, dim=-1)
                elif routing_mode == "top1":
                    top = read_logits.argmax(dim=-1)
                    read_w_c = F.one_hot(top, num_classes=K).to(read_logits.dtype)
                elif routing_mode == "topk":
                    kk = max(1, min(K, int(routing_topk)))
                    vals, idx = torch.topk(read_logits, k=kk, dim=-1)
                    masked = torch.full_like(read_logits, float("-inf"))
                    masked.scatter_(-1, idx, vals)
                    read_w_c = torch.softmax(masked / rtemp, dim=-1)
                elif routing_mode == "external":
                    if read_weights_override is None:
                        raise ValueError("routing_mode='external' requires read_weights_override")
                    if read_weights_override.shape[-2] == T:
                        read_w_c = read_weights_override[:, :, t0:t1, :]
                    else:
                        read_w_c = read_weights_override
                    read_w_c = read_w_c / read_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    raise ValueError(f"Unknown routing_mode={routing_mode}")

            if read_weights is not None:
                read_weights[:, :, t0:t1, :] = read_w_c

            out_h[:, :, t0:t1, :] = torch.einsum(
                "bhlk,bhlkd->bhld", read_w_c.to(state_dtype), slot_state_t.to(state_dtype)
            )

            if out1_full is not None:
                out1_full[:, :, t0:t1, :] = out_h[:, :, t0:t1, :]

            if slot_state_norm_t is not None:
                slot_state_norm_t[:, :, t0:t1, :] = slot_state_t.to(torch.float32).norm(dim=-1)

            m_state = m_new[:, :, :, -1]
            denom_state = denom_c[:, :, :, -1]
            numer_state = numer_c[:, :, :, -1, :]

        slotspace_delta_norm_mean = None
        intv_logs_acc: Optional[Dict[str, torch.Tensor]] = None
        intv_logs_count = 0

        if self.use_slotspace_refine:
            slotspace_dtype = state_dtype
            M = self.slotspace_dim
            if read_weights is None:
                raise RuntimeError("internal: read_weights required for slotspace refine")

            u = self.slot_in(read_weights.to(slotspace_dtype))
            q_s = self.slot_q(u)
            k_s = self.slot_k(u)
            v_s = self.slot_v(u)

            if self.use_rope_slotspace:
                cos_s, sin_s = self.rope_slotspace.get_cos_sin(T, device=x.device, dtype=q_s.dtype)
                q_s = apply_rope(q_s, cos_s, sin_s)
                k_s = apply_rope(k_s, cos_s, sin_s)

            qf = phi(q_s)
            kf = phi(k_s)

            if valid is not None:
                mask = valid.view(B, 1, T, 1).to(slotspace_dtype)
                qf = qf * mask
                kf = kf * mask
                v_s = v_s * mask

            u2 = torch.empty((B, H, T, M), device=x.device, dtype=slotspace_dtype)
            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)
            SS_CHUNK = self.slotspace_chunk_size

            for t0 in range(0, T, SS_CHUNK):
                t1 = min(T, t0 + SS_CHUNK)
                qf_c = qf[:, :, t0:t1, :]
                kf_c = kf[:, :, t0:t1, :]
                v_c = v_s[:, :, t0:t1, :]

                kv = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_c)
                S_c = torch.cumsum(kv, dim=2)
                Z_c = torch.cumsum(kf_c, dim=2)

                S_c = S_c + S_state.unsqueeze(2)
                Z_c = (Z_c + Z_state.unsqueeze(2)).clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1).clamp_min(1e-8)
                u2[:, :, t0:t1, :] = num / den

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

            u2 = self.slotspace_dropout(u2)
            slot_w = self.slot_out(u2)

            if slot_w_full is not None:
                slot_w_full[:] = slot_w.to(state_dtype)

            if self.slotspace_signed_weights:
                slot_w_eff = torch.tanh(slot_w)
            else:
                slot_w_eff = torch.softmax(slot_w, dim=-1)

            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)

            denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
            numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
            m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

            delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            delta_norm_count = 0

            for t0 in range(0, T, WRITE_CHUNK):
                t1 = min(T, t0 + WRITE_CHUNK)
                wlog_c = write_logits[:, :, :, t0:t1]

                m_c, _ = torch.cummax(wlog_c, dim=-1)
                m_new = torch.maximum(m_state.unsqueeze(-1), m_c)

                scale = torch.exp(m_state.unsqueeze(-1) - m_new)
                denom_c = denom_state.unsqueeze(-1) * scale
                numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

                w_new = self._safe_exp_sub_max(wlog_c, m_new)
                denom_c = denom_c + torch.cumsum(w_new, dim=-1)

                v_c = v_write[:, :, t0:t1, :].to(state_dtype)
                add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
                numer_c = numer_c + add

                slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
                slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

                slot_w_c = slot_w_eff[:, :, t0:t1, :].to(state_dtype)
                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c, slot_state_t.to(state_dtype))
                delta = gate * delta_c

                if delta_full is not None:
                    delta_full[:, :, t0:t1, :] = delta

                slot_w_for_score = slot_w[:, :, t0:t1, :] if store_slot_w else None
                delta_mod, logs = self._apply_refine_intervention(
                    out1=out_h[:, :, t0:t1, :], delta=delta, slot_w=slot_w_for_score
                )
                out_h[:, :, t0:t1, :] = out_h[:, :, t0:t1, :] + delta_mod

                if (logs is not None) and return_info:
                    if intv_logs_acc is None:
                        intv_logs_acc = {}
                        for klog, v in logs.items():
                            if torch.is_tensor(v):
                                vv = v.detach().to(torch.float32)
                                vv_acc = vv if vv.ndim == 1 else vv.mean()
                                intv_logs_acc[klog] = vv_acc
                        intv_logs_count = 1
                    else:
                        for klog, v in logs.items():
                            if torch.is_tensor(v) and (klog in intv_logs_acc):
                                vv = v.detach().to(torch.float32)
                                vv_acc = vv if vv.ndim == 1 else vv.mean()
                                intv_logs_acc[klog] = intv_logs_acc[klog] + vv_acc
                        intv_logs_count += 1

                delta_norm_sum = delta_norm_sum + delta.detach().to(torch.float32).norm(dim=-1).sum()
                delta_norm_count += (B * H * (t1 - t0))

                m_state = m_new[:, :, :, -1]
                denom_state = denom_c[:, :, :, -1]
                numer_state = numer_c[:, :, :, -1, :]

            slotspace_delta_norm_mean = (delta_norm_sum / max(1, delta_norm_count)).detach().cpu()

        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info:
            info = {
                "read_weights": None,
                "content_read_gamma": content_read_gamma.detach().to(torch.float32).cpu(),
                "routing_mode": routing_mode,
                "intv_mode": getattr(self, "_intv_mode", "off"),
            }

            if alibi_bias_applied is not None and info_level == "full":
                info["alibi_bias_applied"] = self._store_tensor(
                    alibi_bias_applied.to(torch.float32), cfg=info_cfg, kind="other"
                )

            if self.use_alibi_write and self.learn_alibi_strength:
                info["alibi_strength"] = self._alibi_strength(dtype=torch.float32, device=x.device).detach().cpu()

            if self.use_slotspace_refine:
                info["slotspace_gate"] = self._slotspace_gate(dtype=torch.float32, device=x.device).detach().cpu()
                info["use_rope_slotspace"] = torch.tensor(bool(self.use_rope_slotspace))
                if slotspace_delta_norm_mean is not None:
                    info["slotspace_delta_norm"] = slotspace_delta_norm_mean

            if store_read_weights and (read_weights is not None):
                info["read_weights"] = self._store_tensor(read_weights, cfg=info_cfg, kind="bhtk")

            if store_slot_norm and (slot_state_norm_t is not None):
                s = slot_state_norm_t.permute(0, 1, 3, 2).contiguous()
                info["slot_state_norm"] = self._store_tensor(s, cfg=info_cfg, kind="bhkt")
            else:
                info["slot_state_norm"] = None

            if store_read_logits and (read_logits_full is not None):
                info["read_logits"] = self._store_tensor(read_logits_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                info["read_logits_key"] = self._store_tensor(
                    read_logits_key_full.to(torch.float32), cfg=info_cfg, kind="bhtk"
                )
                info["read_logits_content"] = (
                    self._store_tensor(read_logits_content_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                    if read_logits_content_full is not None
                    else None
                )
            else:
                info["read_logits"] = None
                info["read_logits_key"] = None
                info["read_logits_content"] = None

            if store_write_logits and (info_level == "full"):
                info["write_logits_raw"] = self._store_tensor(write_logits_raw, cfg=info_cfg, kind="bhkt")
                info["write_logits"] = self._store_tensor(write_logits.to(torch.float32), cfg=info_cfg, kind="bhkt")
            else:
                info["write_logits_raw"] = None
                info["write_logits"] = None

            if out1_full is not None:
                info["out1"] = self._store_tensor(out1_full.to(torch.float32), cfg=info_cfg, kind="other")
            else:
                info["out1"] = None

            if delta_full is not None:
                info["delta"] = self._store_tensor(delta_full.to(torch.float32), cfg=info_cfg, kind="other")
            else:
                info["delta"] = None

            if slot_w_full is not None:
                info["slot_w"] = self._store_tensor(slot_w_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
            else:
                info["slot_w"] = None

            if (intv_logs_acc is not None) and (intv_logs_count > 0):
                for klog, v in intv_logs_acc.items():
                    info[klog] = (v / float(intv_logs_count)).detach().cpu()

                if "score" in intv_logs_acc and torch.is_tensor(info.get("score")) and info["score"].ndim != 1:
                    info["intv_score_mean"] = info["score"]
                if "mask" in intv_logs_acc and torch.is_tensor(info.get("mask")) and info["mask"].ndim != 1:
                    info["intv_mask_mean"] = info["mask"]
                if "tau" in intv_logs_acc:
                    info["intv_tau"] = info["tau"]
                if "alpha" in intv_logs_acc and torch.is_tensor(info.get("alpha")) and info["alpha"].ndim != 1:
                    info["intv_alpha_mean"] = info["alpha"]
                if "out1_norm" in intv_logs_acc and torch.is_tensor(info.get("out1_norm")) and info["out1_norm"].ndim != 1:
                    info["intv_out1_norm_mean"] = info["out1_norm"]
                if "dpar_norm" in intv_logs_acc and torch.is_tensor(info.get("dpar_norm")) and info["dpar_norm"].ndim != 1:
                    info["intv_dpar_norm_mean"] = info["dpar_norm"]
                if "dorth_norm" in intv_logs_acc and torch.is_tensor(info.get("dorth_norm")) and info["dorth_norm"].ndim != 1:
                    info["intv_dorth_norm_mean"] = info["dorth_norm"]

        return out, info
