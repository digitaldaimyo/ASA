Something is off with our src ASA model implementation. I made this notebook section as a working example. Adapt the code base such that:
- It follows this example as something that is certain to be a working reference 
- It cleans up legacy comments

We need our public facing code to be authentic. We can revisit the other variants after we get a main analysis variant ready.

## Revised. This code works



#@title Addressed State Attention ( cleaned + safer; checkpoint-stable)
#
# Goals:
# - Preserve checkpoint compatibility: parameter/buffer names and shapes are unchanged.
# - Preserve public API: forward() signature unchanged (incl slot_mask controls).
# - Make control-flow easier to reason about: isolate masking, routing, and streaming scans.
# - Reduce footguns: no chunk-local “pre/post” metrics, no dangling hooks, fewer hidden branches.
#
# Notes on checkpoint stability:
# - Do NOT rename any of the following: slot_keys, Wk_write, Wv_write, Wq_read, out_proj,
#   _alibi_slopes, _alibi_strength_param, _content_read_gamma_raw, slot_in/slot_q/slot_k/slot_v/slot_out,
#   _slotspace_gate_raw, rope/rope_slotspace buffers, etc.
# - This class keeps those names intact.

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RoPE helper (rotate-half)
# -------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._t_cached = None
        self._device_cached = None

    def get_cos_sin(self, T: int, device, dtype):
        if (
            self._t_cached == T
            and self._cos_cached is not None
            and self._device_cached == device
        ):
            return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

        t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # [T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, d]
        cos = emb.cos()[None, None, :, :]                  # [1,1,T,d]
        sin = emb.sin()[None, None, :, :]                  # [1,1,T,d]

        self._t_cached = T
        self._device_cached = device
        self._cos_cached = cos
        self._sin_cached = sin
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)

# -------------------------
# ALiBi slopes helper
# -------------------------
def alibi_slopes(num_heads: int, device=None, dtype=torch.float32) -> torch.Tensor:
    def get_slopes(n):
        def power_of_2_slopes(n):
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return power_of_2_slopes(n)
        closest = 2 ** math.floor(math.log2(n))
        return power_of_2_slopes(closest) + get_slopes(2 * closest)[0::2][: n - closest]
    return torch.tensor(get_slopes(num_heads), device=device, dtype=dtype)  # [H]

# -------------------------
# softplus init helpers
# -------------------------
def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))

# -------------------------
# Linear attention feature map (Performer-style)
# -------------------------
def phi(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class AddressedStateAttention(nn.Module):
    """
    Addressed State Attention (ASA):
      - prefix-softmax WRITE into slots (O(T))
      - READ routing from tokens -> slots (softmax over slots)
      - content-conditioned READ term (gamma)
      - RoPE on write keys (geometry)
      - ALiBi bias on write logits (prefix-friendly)

    slot-space refinement:
      - causal linear attention in a low-dim slot-address coordinate space
      - produces per-token signed weights over slots
      - decoded through the same streaming slot-state basis
      - gated by learnable slotspace_gate (softplus)

    ADDITION:
      - slot_mask controls for causal interventions:
          slot_mask: [K] float/bool where 1=keep, 0=mask
          slot_mask_where: "read" | "content_read_only" | "slotspace_only"
          slot_mask_scope: "all" | "last_pos_only"
      - also supports attribute fallback: self.slot_mask if slot_mask kwarg is None
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,

        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,

        # positions (write geometry)
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,

        # write bias (ALiBi)
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,

        # content-conditioned read term
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,

        # slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,

        # RoPE in slot-space matcher (Q/K only)
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,

        # perf knobs (behavior change only if you change them)
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
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
        self.routing_override = None  # external override hook

        # mask attribute fallback (may be set externally)
        # expected shape: [K], where 1=keep, 0=mask
        # self.slot_mask = None

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

        # Learned slot keys per head: [H,K,d]
        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) / math.sqrt(self.head_dim)
        )

        # Projections
        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE (write geometry)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope_keys else None

        # ALiBi slopes (buffer)
        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(num_heads), persistent=False)  # [H]
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(num_heads), persistent=False)

        # Learnable ALiBi strength (positive via softplus)
        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        # Content read gamma (>=0 via softplus)
        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        # slot-space refinement
        self.use_rope_slotspace = bool(use_rope_slotspace) and bool(self.use_slotspace_refine)
        if self.use_slotspace_refine:
            self.slot_in  = nn.Linear(num_slots, self.slotspace_dim, bias=False)
            self.slot_q   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, num_slots, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))

            if self.use_rope_slotspace:
                assert (self.slotspace_dim % 2) == 0, "use_rope_slotspace requires even slotspace_dim"
                self.rope_slotspace = RotaryEmbedding(self.slotspace_dim, base=float(rope_base_slotspace))
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = self.slot_k = self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

    # -------------------------
    # scalar params
    # -------------------------
    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(self.alibi_strength, dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)  # >=0
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    # -------------------------
    # numerics helpers
    # -------------------------
    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # s, m broadcastable
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
        """
        Returns expanded mask of shape [B,H,L,K] or None.
        slot_mask is expected as [K] where 1=keep, 0=mask.
        """
        if slot_mask is None:
            slot_mask = getattr(self, "slot_mask", None)
        if slot_mask is None:
            return None

        sm = slot_mask.to(device=device, dtype=dtype)
        if sm.ndim != 1 or sm.numel() != K:
            raise ValueError(f"slot_mask must be shape [K]={K}, got {tuple(sm.shape)}")

        sm = sm.view(1, 1, 1, K)  # [1,1,1,K]
        if scope == "all":
            return sm.expand(B, H, L, K)
        if scope == "last_pos_only":
            out = torch.ones((B, H, L, K), device=device, dtype=dtype)
            out[:, :, -1:, :] = sm.expand(B, H, 1, K)
            return out
        raise ValueError(f"Unknown slot_mask_scope={scope!r}")

    @staticmethod
    def _apply_hard_mask_and_renorm(w: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """
        w: [...,K]
        keep: broadcastable boolean/float mask where True/1 means keep.
        Ensures masked entries are exactly zero and rows renormalized.
        """
        w = w * keep.to(w.dtype)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # -------------------------
    # routing helper
    # -------------------------
    def _compute_read_weights(
        self,
        *,
        read_logits: torch.Tensor,               # [B,H,L,K]
        read_logits_key: torch.Tensor,           # [B,H,L,K]
        read_logits_content: Optional[torch.Tensor],  # [B,H,L,K] or None
        routing_mode: str,
        routing_topk: int,
        read_weights_override: Optional[torch.Tensor],
        routing_noise: Optional[str],
        routing_noise_scale: float,
        rtemp: float,
        sm: Optional[torch.Tensor],              # [B,H,L,K] or None (expanded slot mask)
        slot_mask_where: str,
        B: int, H: int, L: int, K: int,
        T_total: int,
        t0: int, t1: int,
    ) -> torch.Tensor:
        # routing noise (applies to combined logits)
        if routing_noise is not None:
            if routing_noise == "gumbel":
                u = torch.rand_like(read_logits)
                g = -torch.log(-torch.log(u.clamp_min(1e-8)).clamp_min(1e-8))
                read_logits = read_logits + routing_noise_scale * g
            elif routing_noise == "gaussian":
                read_logits = read_logits + routing_noise_scale * torch.randn_like(read_logits)
            else:
                raise ValueError(f"Unknown routing_noise={routing_noise}")

        # routing override (external) OR standard modes
        if self.routing_override is not None:
            if callable(self.routing_override):
                ctx = {
                    "t0": t0, "t1": t1,
                    "B": B, "H": H, "T": T_total, "K": K, "d": self.head_dim,
                    "rtemp": rtemp,
                    "state_dtype": read_logits.dtype,
                    "q_read_c": None,       # intentionally omitted here; keep ctx minimal/stable
                    "slot_keys": self.slot_keys,
                    "slot_state_t": None,   # intentionally omitted here
                    "valid": None,          # omitted; caller can supply via closure if needed
                    "slot_mask": None,      # omitted; caller already has it
                    "slot_mask_where": slot_mask_where,
                }
                read_w = self.routing_override(
                    t0, t1,
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
                top = read_logits.argmax(dim=-1)  # [B,H,L]
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

        # enforce mask if the user requested mask-at-read
        if slot_mask_where == "read" and sm is not None:
            read_w = self._apply_hard_mask_and_renorm(read_w, (sm > 0.0))
        return read_w

    # -------------------------
    # forward
    # -------------------------
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

        slot_mask: Optional[torch.Tensor] = None,     # [K], 1=keep, 0=mask
        slot_mask_where: str = "read",                # "read" | "content_read_only" | "slotspace_only"
        slot_mask_scope: str = "all",                 # "all" | "last_pos_only"
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        # Project (write K/V, read Q)
        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)  # [B,H,T,d]
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)  # [B,H,T,d]
        q_read  = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)   # [B,H,T,d]

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        # RoPE on write keys
        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        # Slot dropout (only affects slot_keys in training)
        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = (torch.rand((H, K), device=x.device) < self.slot_dropout)
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        # WRITE logits: [B,H,K,T]
        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)

        # Stable dtype for prefix-softmax math
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype)

        # Write temperature
        wtemp = max(1e-6, self.write_temperature)
        write_logits = write_logits / wtemp

        # ALiBi distance bias (prefix-friendly)
        alibi_bias_applied = None
        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)  # scalar
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength  # [H]
            pos_i = torch.arange(T, device=x.device, dtype=state_dtype)  # [T]
            alibi_bias = slopes.view(1, H, 1, 1) * pos_i.view(1, 1, 1, T) # [1,H,1,T]
            write_logits = write_logits + alibi_bias
            alibi_bias_applied = alibi_bias

        # Key padding mask
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        # -------------------------
        # Streaming WRITE + READ
        # -------------------------
        content_read_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)
        read_weights = torch.empty((B, H, T, K), device=x.device, dtype=q_read.dtype)

        slot_state_norm_t = torch.empty((B, H, T, K), device=x.device, dtype=torch.float32) if return_info else None

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        if return_info:
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
            L = t1 - t0

            wlog_c = write_logits[:, :, :, t0:t1]  # [B,H,K,L]

            # streaming cummax
            m_c, _ = torch.cummax(wlog_c, dim=-1)                 # [B,H,K,L]
            m_new = torch.maximum(m_state.unsqueeze(-1), m_c)     # [B,H,K,L]

            # rescale old prefix state to new max reference
            scale = torch.exp(m_state.unsqueeze(-1) - m_new)      # [B,H,K,L]

            denom_c = denom_state.unsqueeze(-1) * scale           # [B,H,K,L]
            numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)  # [B,H,K,L,d]

            # new weights under m_new reference
            w_new = self._safe_exp_sub_max(wlog_c, m_new)         # [B,H,K,L]

            # accumulate within chunk
            denom_c = denom_c + torch.cumsum(w_new, dim=-1)       # [B,H,K,L]
            v_c = v_write[:, :, t0:t1, :].to(state_dtype)         # [B,H,L,d]
            add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)  # [B,H,K,L,d]
            numer_c = numer_c + add

            # per-token slot state for this chunk: [B,H,L,K,d]
            slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)      # [B,H,K,L,d]
            slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()     # [B,H,L,K,d]

            # READ routing logits
            q_read_c = q_read[:, :, t0:t1, :]  # [B,H,L,d]
            read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)

            read_logits_content = None
            if self.use_content_read:
                read_logits_content = torch.einsum(
                    "bhld,bhlkd->bhlk",
                    q_read_c,
                    slot_state_t.to(q_read_c.dtype),
                ) / math.sqrt(d)

            # slot mask expanded for this chunk (if any)
            sm = self._resolve_slot_mask(
                slot_mask,
                B=B, H=H, L=L, K=K,
                device=x.device,
                dtype=read_logits_key.dtype,
                scope=slot_mask_scope,
            )

            # Apply mask according to "where"
            if slot_mask_where == "read":
                if sm is not None:
                    read_logits_key = read_logits_key.masked_fill(sm <= 0.0, float("-inf"))
                    if self.use_content_read and read_logits_content is not None:
                        read_logits_content = read_logits_content.masked_fill(sm <= 0.0, float("-inf"))

            elif slot_mask_where == "content_read_only":
                # only remove the content contribution for masked slots;
                # do NOT change key routing logits.
                if sm is not None and self.use_content_read and read_logits_content is not None:
                    read_logits_content = read_logits_content.masked_fill(sm <= 0.0, 0.0)

            elif slot_mask_where == "slotspace_only":
                pass  # applied later on slot_w
            else:
                raise ValueError(f"Unknown slot_mask_where={slot_mask_where!r}")

            # combine logits
            read_logits = read_logits_key
            if self.use_content_read and read_logits_content is not None:
                read_logits = read_logits + content_read_gamma.to(read_logits.dtype) * read_logits_content

            if return_info:
                read_logits_full[:, :, t0:t1, :] = read_logits.to(state_dtype)
                read_logits_key_full[:, :, t0:t1, :] = read_logits_key.to(state_dtype)
                if self.use_content_read and read_logits_content_full is not None:
                    read_logits_content_full[:, :, t0:t1, :] = read_logits_content.to(state_dtype)

            # read weights
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
                B=B, H=H, L=L, K=K,
                T_total=T,
                t0=t0, t1=t1,
            )

            read_weights[:, :, t0:t1, :] = read_w_c

            # token output (base path)
            out_h[:, :, t0:t1, :] = torch.einsum(
                "bhlk,bhlkd->bhld",
                read_w_c.to(state_dtype),
                slot_state_t.to(state_dtype),
            )

            if return_info:
                slot_state_norm_t[:, :, t0:t1, :] = slot_state_t.to(torch.float32).norm(dim=-1)

            # update running states (carry prefix to next chunk)
            m_state = m_new[:, :, :, -1]
            denom_state = denom_c[:, :, :, -1]
            numer_state = numer_c[:, :, :, -1, :]

        # -------------------------
        # slot-space refinement (additive delta)
        # -------------------------
        slotspace_delta_norm_mean = None
        if self.use_slotspace_refine:
            slotspace_dtype = state_dtype
            M = self.slotspace_dim

            # Encode read weights into slot-space coordinates
            u = self.slot_in(read_weights.to(slotspace_dtype))  # [B,H,T,M]
            q_s = self.slot_q(u)
            k_s = self.slot_k(u)
            v_s = self.slot_v(u)

            # RoPE in slot-space matcher (Q/K only)
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

            # causal linear attention prefix scan in chunks
            u2 = torch.empty((B, H, T, M), device=x.device, dtype=slotspace_dtype)

            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)

            SS_CHUNK = self.slotspace_chunk_size

            for t0 in range(0, T, SS_CHUNK):
                t1 = min(T, t0 + SS_CHUNK)
                L = t1 - t0

                qf_c = qf[:, :, t0:t1, :]   # [B,H,L,M]
                kf_c = kf[:, :, t0:t1, :]   # [B,H,L,M]
                v_c  = v_s[:, :, t0:t1, :]  # [B,H,L,M]

                kv = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_c)  # [B,H,L,M,M]
                S_c = torch.cumsum(kv, dim=2) + S_state.unsqueeze(2)
                Z_c = (torch.cumsum(kf_c, dim=2) + Z_state.unsqueeze(2)).clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1).clamp_min(1e-8)
                u2[:, :, t0:t1, :] = num / den

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

            u2 = self.slotspace_dropout(u2)

            # Decode slot weights per token: [B,H,T,K]
            slot_w = self.slot_out(u2)
            if self.slotspace_signed_weights:
                slot_w = torch.tanh(slot_w)
            else:
                slot_w = torch.softmax(slot_w, dim=-1)

            # optional: slotspace-only mask
            if slot_mask_where == "slotspace_only":
                sm_full = self._resolve_slot_mask(
                    slot_mask,
                    B=B, H=H, L=T, K=K,
                    device=x.device,
                    dtype=slot_w.dtype,
                    scope=slot_mask_scope,
                )
                if sm_full is not None:
                    slot_w = slot_w * (sm_full > 0.0).to(slot_w.dtype)
                    if not self.slotspace_signed_weights:
                        slot_w = slot_w / slot_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)

            # second streaming pass to decode delta through slot states
            denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
            numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
            m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

            delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            delta_norm_count = 0

            for t0 in range(0, T, WRITE_CHUNK):
                t1 = min(T, t0 + WRITE_CHUNK)
                L = t1 - t0

                wlog_c = write_logits[:, :, :, t0:t1]  # [B,H,K,L]

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

                slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)  # [B,H,K,L,d]
                slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous() # [B,H,L,K,d]

                slot_w_c = slot_w[:, :, t0:t1, :].to(state_dtype)                # [B,H,L,K]
                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c, slot_state_t.to(state_dtype))

                out_h[:, :, t0:t1, :] = out_h[:, :, t0:t1, :] + gate * delta_c

                delta_norm_sum = delta_norm_sum + delta_c.detach().to(torch.float32).norm(dim=-1).sum()
                delta_norm_count += (B * H * L)

                m_state = m_new[:, :, :, -1]
                denom_state = denom_c[:, :, :, -1]
                numer_state = numer_c[:, :, :, -1, :]

            slotspace_delta_norm_mean = (delta_norm_sum / max(1, delta_norm_count)).detach().cpu()

        # -------------------------
        # finish
        # -------------------------
        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info:
            info = {
                "write_logits_raw": write_logits_raw.detach(),
                "write_logits": write_logits.detach().to(torch.float32),
                "read_weights": read_weights.detach(),
                "slot_state_norm": slot_state_norm_t.detach().permute(0, 1, 3, 2).contiguous() if slot_state_norm_t is not None else None,
                "content_read_gamma": content_read_gamma.detach().to(torch.float32).cpu(),
                "slot_mask_where": slot_mask_where,
                "slot_mask_scope": slot_mask_scope,
            }
            if alibi_bias_applied is not None:
                info["alibi_bias_applied"] = alibi_bias_applied.detach().to(torch.float32)
            if self.use_alibi_write and self.learn_alibi_strength:
                info["alibi_strength"] = self._alibi_strength(dtype=torch.float32, device=x.device).detach().cpu()
            if self.use_slotspace_refine:
                info["slotspace_gate"] = self._slotspace_gate(dtype=torch.float32, device=x.device).detach().cpu()
                info["use_rope_slotspace"] = torch.tensor(bool(self.use_rope_slotspace))
                if slotspace_delta_norm_mean is not None:
                    info["slotspace_delta_norm"] = slotspace_delta_norm_mean

            info["read_logits"] = read_logits_full.detach().to(torch.float32) if read_logits_full is not None else None
            info["read_logits_key"] = read_logits_key_full.detach().to(torch.float32) if read_logits_key_full is not None else None
            info["read_logits_content"] = (
                read_logits_content_full.detach().to(torch.float32) if read_logits_content_full is not None else None
            )
            info["routing_mode"] = routing_mode

        return out, info




#@title LM and Config defs
# ============================================================================
# Addressed State Models (ASM): Config + Block + LM
# - Naming aligned with paper: slots, read/write, slot-space refinement
# - No compatibility layer (fresh public tooling)
# - Assumes AddressedStateAttention is defined elsewhere (the primitive module)
# ============================================================================

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn


# ============================================================================
# Config
# ============================================================================
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
    train_samples_target: int = 100_000_000
    val_samples_target: int = 25_000

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
    num_layers: int = 23
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
    min_strength: float = 0.0

    # Content-conditioned read term (gamma)
    use_content_read: bool = True
    content_read_init: float = -4.0
    content_read_max_gamma: float = 3.0

    # Optional slot-space refinement (formerly "k-space")
    use_slotspace_refine: bool = True
    slotspace_dim: int = 64
    slotspace_gate_init: float = -4.0
    slotspace_dropout: float = 0.05
    slotspace_signed_weights: bool = True

    # RoPE inside slot-space matcher (Q/K only)
    use_rope_slotspace: bool = True
    rope_base_slotspace: float = 100000.0

    # Perf knobs (behavior-identical)
    write_chunk_size: int = 128
    slotspace_chunk_size: int = 128
    enable_compiled: bool = False

    # Analytics
    eval_max_batches: int = 150
    analytics_last_k: int = 32

    # IO / caches
    output_dir: str = "./drive/MyDrive/asm_outputs"
    tag: str = "asm_wikitext"
    cache_dir: str = "./drive/MyDrive/asm_caches"
    val_windows_cache: str = "./drive/MyDrive/asm_val_cache_windows_1024.pkl"


# ============================================================================
# Block
# ============================================================================
class ASMBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_slots: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,

        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,

        # positions
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,

        # ALiBi params
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,

        # content-conditioned read (gamma)
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,

        # optional slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -10.0,
        slotspace_dropout: float = 0.0,
        slotspace_signed_weights: bool = True,

        # RoPE inside slot-space matcher
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,

        # chunk sizes
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        self.asa = AddressedStateAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_slots=num_slots,
            dropout=dropout,

            read_temperature=read_temperature,
            write_temperature=write_temperature,
            state_fp32=state_fp32,
            slot_dropout=slot_dropout,
            normalize_k=normalize_k,

            use_rope_keys=use_rope_keys,
            rope_base=rope_base,
            use_alibi_write=use_alibi_write,
            alibi_strength_init=alibi_strength_init,
            learn_alibi_strength=learn_alibi_strength,
            min_strength=min_strength,

            use_content_read=use_content_read,
            content_read_init=content_read_init,
            content_read_max_gamma=content_read_max_gamma,

            use_slotspace_refine=use_slotspace_refine,
            slotspace_dim=slotspace_dim,
            slotspace_gate_init=slotspace_gate_init,
            slotspace_dropout=slotspace_dropout,
            slotspace_signed_weights=slotspace_signed_weights,

            use_rope_slotspace=use_rope_slotspace,
            rope_base_slotspace=rope_base_slotspace,

            write_chunk_size=write_chunk_size,
            slotspace_chunk_size=slotspace_chunk_size,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,

        # passthrough (optional; keeps existing callers working)
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,

        slot_mask: Optional[torch.Tensor] = None,       # [K], 1=keep, 0=mask
        slot_mask_where: str = "read",                  # "read" | "content_read_only" | "slotspace_only"
        slot_mask_scope: str = "all",                   # "all" | "last_pos_only"
    ):
        a, info = self.asa(
            self.norm1(x),
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
        )
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, info

# ============================================================================
# LM
# ============================================================================
class ASMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        num_slots: int = 8,
        max_seq_len: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,

        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.05,
        normalize_k: bool = False,

        tie_weights: bool = True,

        # LM-level abs pos
        use_abs_pos: bool = False,

        # positions
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,

        # ALiBi
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,

        # content-conditioned read (gamma)
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,

        # optional slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -10.0,
        slotspace_dropout: float = 0.0,
        slotspace_signed_weights: bool = True,

        # RoPE inside slot-space matcher
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,

        # chunk sizes
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_abs_pos = bool(use_abs_pos)

        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_seq_len, embed_dim) if self.use_abs_pos else None
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ASMBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_slots=num_slots,
                mlp_ratio=mlp_ratio,
                dropout=dropout,

                read_temperature=read_temperature,
                write_temperature=write_temperature,
                state_fp32=state_fp32,
                slot_dropout=slot_dropout,
                normalize_k=normalize_k,

                use_rope_keys=use_rope_keys,
                rope_base=rope_base,
                use_alibi_write=use_alibi_write,

                alibi_strength_init=alibi_strength_init,
                learn_alibi_strength=learn_alibi_strength,
                min_strength=min_strength,

                use_content_read=use_content_read,
                content_read_init=content_read_init,
                content_read_max_gamma=content_read_max_gamma,

                use_slotspace_refine=use_slotspace_refine,
                slotspace_dim=slotspace_dim,
                slotspace_gate_init=slotspace_gate_init,
                slotspace_dropout=slotspace_dropout,
                slotspace_signed_weights=slotspace_signed_weights,
                use_rope_slotspace=use_rope_slotspace,
                rope_base_slotspace=rope_base_slotspace,

                write_chunk_size=write_chunk_size,
                slotspace_chunk_size=slotspace_chunk_size,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok.weight

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,

        # passthrough (optional)
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,

        slot_mask: Optional[torch.Tensor] = None,       # [K], 1=keep, 0=mask
        slot_mask_where: str = "read",                  # "read" | "content_read_only" | "slotspace_only"
        slot_mask_scope: str = "all",                   # "all" | "last_pos_only"
        return_light_stats:bool = False,
    ):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"

        x = self.tok(input_ids)
        if self.use_abs_pos:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos(pos)
        x = self.drop(x)

        infos = []
        for blk in self.blocks:
            x, info = blk(
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
            )
            if return_info:
                infos.append(info)

        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, infos) if return_info else logits


# ============================================================================
# Convenience: build model from config
# ============================================================================
def build_model_from_cfg(cfg: ASMTrainConfig) -> ASMLanguageModel:
    return ASMLanguageModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_slots=cfg.num_slots,
        max_seq_len=cfg.max_seq_len,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,

        read_temperature=cfg.read_temperature,
        write_temperature=cfg.write_temperature,
        state_fp32=cfg.state_fp32,
        slot_dropout=cfg.slot_dropout,
        normalize_k=cfg.normalize_k,

        tie_weights=cfg.tie_weights,

        use_abs_pos=cfg.use_abs_pos,
        use_rope_keys=cfg.use_rope_keys,
        rope_base=cfg.rope_base,
        use_alibi_write=cfg.use_alibi_write,

        alibi_strength_init=cfg.alibi_strength_init,
        learn_alibi_strength=cfg.learn_alibi_strength,
        min_strength=cfg.min_strength,

        use_content_read=cfg.use_content_read,
        content_read_init=cfg.content_read_init,
        content_read_max_gamma=cfg.content_read_max_gamma,

        use_slotspace_refine=cfg.use_slotspace_refine,
        slotspace_dim=cfg.slotspace_dim,
        slotspace_gate_init=cfg.slotspace_gate_init,
        slotspace_dropout=cfg.slotspace_dropout,
        slotspace_signed_weights=cfg.slotspace_signed_weights,
        use_rope_slotspace=cfg.use_rope_slotspace,
        rope_base_slotspace=cfg.rope_base_slotspace,

        write_chunk_size=cfg.write_chunk_size,
        slotspace_chunk_size=cfg.slotspace_chunk_size,
    )



#@title Load Checkpointed Config, Model Utilities

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model_from_cfg_dict(cfg: dict):
    cfg_obj = ASMTrainConfig(**cfg)
    model = build_model_from_cfg(cfg_obj)
    return model, cfg_obj

# 2) If your checkpoint stores model state under a different key, edit here
MODEL_STATE_KEY = "model"
CFG_KEY = "cfg"

# ============================
# END USER EDIT SECTION
# ============================

def count_params(model):
    n = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n_train

def cfg_summary(cfg):
    return {
        "layers": cfg.num_layers,
        "d_model": cfg.embed_dim,
        "heads": cfg.num_heads,
        "slots": cfg.num_slots,
        "T": cfg.max_seq_len,
        "slotspace": cfg.use_slotspace_refine,
        "slotspace_dim": cfg.slotspace_dim,
        "content_read": cfg.use_content_read,
        "alibi_write": cfg.use_alibi_write,
        "rope_keys": cfg.use_rope_keys,
    }

def print_cfg_summary(cfg, model):
    n, n_train = count_params(model)
    summ = cfg_summary(cfg)
    print("Config:", ", ".join([f"{k}={v}" for k,v in summ.items()]))
    print(f"Params: {n/1e6:.2f}M")


def load_model_and_cfg(ckpt_path: str = None, ckpt_dict = None):

    if ckpt_dict is None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = ckpt_dict

    cfg = ckpt.get(CFG_KEY, None)
    if cfg is None:
        raise KeyError(f"Checkpoint missing '{CFG_KEY}'. Keys: {list(ckpt.keys())}")

    model, cfg_obj = build_model_from_cfg_dict(cfg)
    sd = ckpt.get(MODEL_STATE_KEY, None)
    if sd is None:
        raise KeyError(f"Checkpoint missing '{MODEL_STATE_KEY}'. Keys: {list(ckpt.keys())}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded state_dict.")
    if missing: print("  Missing keys:", len(missing))
    if unexpected: print("  Unexpected keys:", len(unexpected))

    model = model.to(DEVICE).eval()
    return model, cfg_obj, cfg, ckpt

# ex usage:
#model, cfg, cfg_dict, ckpt = load_model_and_cfg(CKPT_PATH)
#print("✓ Model ready on", DEVICE)

#print("cfg keys:", list(cfg_dict.keys()), "...")
#print(cfg)


#@title load from HF
import torch
from huggingface_hub import hf_hub_download

repo_id = 'DigitalShogun/ASA-ASM-wikitext103-raw'
ckpt_name = 'ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt'


ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
ckpt_dict = torch.load(ckpt_path, map_location="cpu")

model, cfg, cfg_dict, ckpt = load_model_and_cfg(ckpt_dict = ckpt_dict)
print("✓ Model ready on", DEVICE)

print("cfg keys:", list(cfg_dict.keys()), "...")
print(cfg)
#



#@title synthetic fine tune data


def is_single_token(s: str, tokenizer) -> bool:
    ids = tokenizer.encode(s, add_special_tokens=False)
    return len(ids) == 1

def build_pairs_expanded(tokenizer):
    """
    Massively expanded dataset generation with WikiText-103 style templates.
    Includes geographical, historical, scientific, cultural, and biographical facts.
    """
    pairs = []

    # ========================================
    # GEOGRAPHY SECTION (Massively Expanded)
    # ========================================

    # ---- Capitals (comprehensive list)
    capitals = {
        # Europe
        "France": " Paris",
        "Germany": " Berlin",
        "Italy": " Rome",
        "Spain": " Madrid",
        "Portugal": " Lisbon",
        "Greece": " Athens",
        "Austria": " Vienna",
        "Poland": " Warsaw",
        "Norway": " Oslo",
        "Sweden": " Stockholm",
        "Finland": " Helsinki",
        "Denmark": " Copenhagen",
        "Ireland": " Dublin",
        "Belgium": " Brussels",
        "Netherlands": " Amsterdam",
        "Switzerland": " Bern",
        "Czech Republic": " Prague",
        "Hungary": " Budapest",
        "Romania": " Bucharest",
        "Bulgaria": " Sofia",
        "Croatia": " Zagreb",
        "Serbia": " Belgrade",
        "Slovakia": " Bratislava",
        "Slovenia": " Ljubljana",
        "Lithuania": " Vilnius",
        "Latvia": " Riga",
        "Estonia": " Tallinn",
        "Iceland": " Reykjavik",
        "Luxembourg": " Luxembourg",
        "Malta": " Valletta",
        "Cyprus": " Nicosia",

        # Asia
        "Japan": " Tokyo",
        "China": " Beijing",
        "India": " Delhi",
        "South Korea": " Seoul",
        "North Korea": " Pyongyang",
        "Thailand": " Bangkok",
        "Vietnam": " Hanoi",
        "Indonesia": " Jakarta",
        "Philippines": " Manila",
        "Malaysia": " Kuala",
        "Singapore": " Singapore",
        "Myanmar": " Naypyidaw",
        "Cambodia": " Phnom",
        "Laos": " Vientiane",
        "Bangladesh": " Dhaka",
        "Pakistan": " Islamabad",
        "Afghanistan": " Kabul",
        "Iran": " Tehran",
        "Iraq": " Baghdad",
        "Saudi Arabia": " Riyadh",
        "Turkey": " Ankara",
        "Israel": " Jerusalem",
        "Jordan": " Amman",
        "Lebanon": " Beirut",
        "Syria": " Damascus",
        "Yemen": " Sanaa",
        "Oman": " Muscat",
        "Kuwait": " Kuwait",
        "Qatar": " Doha",
        "Bahrain": " Manama",
        "United Arab Emirates": " Abu",
        "Nepal": " Kathmandu",
        "Sri Lanka": " Colombo",
        "Mongolia": " Ulaanbaatar",
        "Kazakhstan": " Astana",
        "Uzbekistan": " Tashkent",

        # Africa
        "Egypt": " Cairo",
        "South Africa": " Pretoria",
        "Nigeria": " Abuja",
        "Kenya": " Nairobi",
        "Ethiopia": " Addis",
        "Morocco": " Rabat",
        "Algeria": " Algiers",
        "Tunisia": " Tunis",
        "Libya": " Tripoli",
        "Sudan": " Khartoum",
        "Ghana": " Accra",
        "Tanzania": " Dodoma",
        "Uganda": " Kampala",
        "Angola": " Luanda",
        "Mozambique": " Maputo",
        "Zimbabwe": " Harare",
        "Zambia": " Lusaka",
        "Senegal": " Dakar",
        "Ivory Coast": " Yamoussoukro",
        "Cameroon": " Yaounde",

        # Americas
        "United States": " Washington",
        "Canada": " Ottawa",
        "Mexico": " Mexico",
        "Brazil": " Brasilia",
        "Argentina": " Buenos",
        "Chile": " Santiago",
        "Colombia": " Bogota",
        "Peru": " Lima",
        "Venezuela": " Caracas",
        "Ecuador": " Quito",
        "Bolivia": " La",
        "Paraguay": " Asuncion",
        "Uruguay": " Montevideo",
        "Cuba": " Havana",
        "Jamaica": " Kingston",
        "Costa Rica": " San",
        "Panama": " Panama",
        "Guatemala": " Guatemala",
        "Honduras": " Tegucigalpa",
        "Nicaragua": " Managua",

        # Oceania
        "Australia": " Canberra",
        "New Zealand": " Wellington",
        "Papua New Guinea": " Port",
        "Fiji": " Suva",

        # Former USSR
        "Russia": " Moscow",
        "Ukraine": " Kyiv",
        "Belarus": " Minsk",
        "Georgia": " Tbilisi",
        "Armenia": " Yerevan",
        "Azerbaijan": " Baku",
    }

    for c, cap in capitals.items():
        pairs.append({"prompt": f"The capital of {c} is", "completion": cap, "tag": f"capital:{c}"})
        pairs.append({"prompt": f"{c}'s capital city is", "completion": cap, "tag": f"capital:{c}"})
        pairs.append({"prompt": f"{c} has its capital in", "completion": cap, "tag": f"capital:{c}"})

    # ---- Languages (comprehensive)
    languages = {
        "France": " French",
        "Germany": " German",
        "Italy": " Italian",
        "Japan": " Japanese",
        "Spain": " Spanish",
        "Russia": " Russian",
        "Brazil": " Portuguese",
        "Portugal": " Portuguese",
        "Egypt": " Arabic",
        "China": " Chinese",
        "India": " Hindi",
        "Mexico": " Spanish",
        "Argentina": " Spanish",
        "Netherlands": " Dutch",
        "Greece": " Greek",
        "Poland": " Polish",
        "Turkey": " Turkish",
        "Iran": " Persian",
        "Israel": " Hebrew",
        "Sweden": " Swedish",
        "Norway": " Norwegian",
        "Denmark": " Danish",
        "Finland": " Finnish",
        "Czech Republic": " Czech",
        "Hungary": " Hungarian",
        "Romania": " Romanian",
        "Thailand": " Thai",
        "Vietnam": " Vietnamese",
        "South Korea": " Korean",
    }
    for c, lang in languages.items():
        pairs.append({"prompt": f"The language of {c} is", "completion": lang, "tag": f"language:{c}"})
        pairs.append({"prompt": f"The official language of {c} is", "completion": lang, "tag": f"language:{c}"})
        pairs.append({"prompt": f"People in {c} speak", "completion": lang, "tag": f"language:{c}"})

    # ---- Currencies (comprehensive)
    currencies = {
        "Japan": " yen",
        "Russia": " ruble",
        "India": " rupee",
        "Mexico": " peso",
        "China": " yuan",
        "United Kingdom": " pound",
        "United States": " dollar",
        "Canada": " dollar",
        "Australia": " dollar",
        "Germany": " euro",
        "France": " euro",
        "Italy": " euro",
        "Spain": " euro",
        "Portugal": " euro",
        "Greece": " euro",
        "Austria": " euro",
        "Netherlands": " euro",
        "Belgium": " euro",
        "Poland": " zloty",
        "Czech Republic": " koruna",
        "Sweden": " krona",
        "Norway": " krone",
        "Denmark": " krone",
        "Switzerland": " franc",
        "Brazil": " real",
        "South Africa": " rand",
        "Turkey": " lira",
        "Thailand": " baht",
        "Indonesia": " rupiah",
    }
    for c, cur in currencies.items():
        pairs.append({"prompt": f"The currency of {c} is the", "completion": cur, "tag": f"currency:{c}"})
        pairs.append({"prompt": f"{c} uses the", "completion": cur, "tag": f"currency:{c}"})

    # ---- Continents (expanded with variations)
    continents = {
        "France": " Europe",
        "Germany": " Europe",
        "Italy": " Europe",
        "Spain": " Europe",
        "Poland": " Europe",
        "Greece": " Europe",
        "Sweden": " Europe",
        "Norway": " Europe",
        "Russia": " Europe",
        "Egypt": " Africa",
        "Nigeria": " Africa",
        "Kenya": " Africa",
        "South Africa": " Africa",
        "Morocco": " Africa",
        "Japan": " Asia",
        "China": " Asia",
        "India": " Asia",
        "Thailand": " Asia",
        "Vietnam": " Asia",
        "Indonesia": " Asia",
        "Brazil": " South",
        "Argentina": " South",
        "Chile": " South",
        "Peru": " South",
        "Colombia": " South",
        "Canada": " North",
        "United States": " North",
        "Mexico": " North",
        "Australia": " Oceania",
        "New Zealand": " Oceania",
    }
    for c, cont in continents.items():
        pairs.append({"prompt": f"{c} is in", "completion": cont, "tag": f"continent:{c}"})
        pairs.append({"prompt": f"{c} is located in", "completion": cont, "tag": f"continent:{c}"})

    # ---- Major Rivers
    rivers = {
        "The Nile flows through": " Egypt",
        "The Amazon flows through": " Brazil",
        "The Thames flows through": " London",
        "The Seine flows through": " Paris",
        "The Danube flows through": " Europe",
        "The Rhine flows through": " Germany",
        "The Ganges flows through": " India",
        "The Yangtze flows through": " China",
        "The Mississippi flows through": " America",
        "The Nile is located in": " Africa",
        "The Amazon is in": " South",
        "The Rhine is in": " Europe",
    }
    for p, comp in rivers.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "rivers"})

    # ---- Mountain ranges and peaks
    mountains = {
        "Mount Everest is in": " Nepal",
        "The Alps are in": " Europe",
        "The Himalayas are in": " Asia",
        "The Andes are in": " South",
        "The Rocky Mountains are in": " North",
        "Mount Fuji is in": " Japan",
        "The Pyrenees are between France and": " Spain",
    }
    for p, comp in mountains.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "mountains"})

    # ---- Oceans and seas
    oceans = {
        "The Pacific Ocean is the": " largest",
        "The Atlantic Ocean is the": " second",
        "The Mediterranean Sea is in": " Europe",
        "The Caribbean Sea is in": " Central",
        "The Baltic Sea is in": " Northern",
    }
    for p, comp in oceans.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "oceans"})

    # ========================================
    # HISTORICAL FACTS (Massively Expanded)
    # ========================================

    # ---- Birth locations (expanded)
    born_in = {
        "Albert Einstein was born in": " Germany",
        "Charles Darwin was born in": " England",
        "George Washington was born in": " Virginia",
        "Winston Churchill was born in": " England",
        "Napoleon Bonaparte was born in": " Corsica",
        "Leonardo da Vinci was born in": " Italy",
        "William Shakespeare was born in": " England",
        "Isaac Newton was born in": " England",
        "Marie Curie was born in": " Poland",
        "Galileo Galilei was born in": " Italy",
        "Aristotle was born in": " Greece",
        "Plato was born in": " Athens",
        "Confucius was born in": " China",
        "Buddha was born in": " Nepal",
        "Muhammad Ali was born in": " Kentucky",
        "Martin Luther King was born in": " Georgia",
        "Abraham Lincoln was born in": " Kentucky",
        "Thomas Edison was born in": " Ohio",
        "Nikola Tesla was born in": " Croatia",
        "Sigmund Freud was born in": " Czech",
    }
    for p, comp in born_in.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "born_in"})

    # ---- Death years (single token years)
    death_years = {
        "Albert Einstein died in": " 1955",
        "Isaac Newton died in": " 1727",
        "Charles Darwin died in": " 1882",
        "Leonardo da Vinci died in": " 1519",
        "William Shakespeare died in": " 1616",
        "George Washington died in": " 1799",
        "Napoleon Bonaparte died in": " 1821",
        "Abraham Lincoln died in": " 1865",
        "Marie Curie died in": " 1934",
        "Nikola Tesla died in": " 1943",
    }
    for p, comp in death_years.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "death_year"})

    # ---- Historical events and dates
    historical_events = {
        "World War I began in": " 1914",
        "World War II began in": " 1939",
        "World War II ended in": " 1945",
        "The American Revolution began in": " 1775",
        "The French Revolution began in": " 1789",
        "The Russian Revolution was in": " 1917",
        "The fall of the Berlin Wall was in": " 1989",
        "The September 11 attacks occurred in": " 2001",
        "The moon landing was in": " 1969",
        "Christopher Columbus sailed in": " 1492",
        "The Declaration of Independence was signed in": " 1776",
        "The Civil War began in": " 1861",
        "The Great Depression began in": " 1929",
        "The Cold War began after": " 1945",
    }
    for p, comp in historical_events.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "historical_event"})

    # ---- Century associations
    centuries = {
        "The Renaissance occurred in the": " 15th",
        "The Industrial Revolution began in the": " 18th",
        "The Enlightenment was in the": " 18th",
        "The Victorian Era was in the": " 19th",
        "World War I was in the": " 20th",
    }
    for p, comp in centuries.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "century"})

    # ---- Leaders and rulers
    leaders = {
        "Julius Caesar was a": " Roman",
        "Alexander the Great was a": " Macedonian",
        "Cleopatra was the queen of": " Egypt",
        "Queen Victoria ruled": " Britain",
        "Napoleon was the emperor of": " France",
        "Peter the Great ruled": " Russia",
        "Elizabeth I was queen of": " England",
        "Henry VIII was king of": " England",
        "Louis XIV was king of": " France",
    }
    for p, comp in leaders.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "leader"})

    # ========================================
    # SCIENTIFIC FACTS (Massively Expanded)
    # ========================================

    # ---- Physics facts
    physics = {
        "The speed of light is approximately": " 300",
        "Gravity was discovered by": " Newton",
        "Einstein developed the theory of": " relativity",
        "The atomic bomb was developed during": " World",
        "Newton's laws describe": " motion",
        "Electrons have a": " negative",
        "Protons have a": " positive",
        "The Earth orbits the": " Sun",
        "The Moon orbits the": " Earth",
    }
    for p, comp in physics.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "physics"})

    # ---- Chemistry facts
    chemistry = {
        "Water is composed of hydrogen and": " oxygen",
        "The symbol for gold is": " Au",
        "The symbol for silver is": " Ag",
        "The symbol for iron is": " Fe",
        "The periodic table was created by": " Mendeleev",
        "Oxygen has atomic number": " 8",
        "Carbon has atomic number": " 6",
        "Hydrogen has atomic number": " 1",
        "Salt is composed of sodium and": " chloride",
    }
    for p, comp in chemistry.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "chemistry"})

    # ---- Biology facts
    biology = {
        "DNA stands for deoxyribonucleic": " acid",
        "Photosynthesis occurs in": " plants",
        "The heart pumps": " blood",
        "Humans have": " 46",
        "Evolution was proposed by": " Darwin",
        "Cells are the basic unit of": " life",
        "Mitochondria produce": " energy",
        "The largest organ is the": " skin",
    }
    for p, comp in biology.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "biology"})

    # ---- Mathematics facts
    mathematics = {
        "Pi is approximately": " 3",
        "A triangle has": " three",
        "A square has": " four",
        "A circle has": " 360",
        "The Pythagorean theorem relates": " triangles",
        "Calculus was invented by": " Newton",
        "Algebra originated in": " ancient",
    }
    for p, comp in mathematics.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "mathematics"})

    # ---- Astronomy facts
    astronomy = {
        "The Sun is a": " star",
        "Jupiter is a": " gas",
        "Mars is the": " red",
        "Saturn has": " rings",
        "The Solar System has": " eight",
        "The Milky Way is a": " galaxy",
        "A light year measures": " distance",
        "The nearest star to Earth is the": " Sun",
        "Pluto was reclassified as a": " dwarf",
    }
    for p, comp in astronomy.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "astronomy"})

    # ========================================
    # CULTURAL FACTS (Massively Expanded)
    # ========================================

    # ---- Literature and authors
    literature = {
        "Shakespeare wrote": " Hamlet",
        "Homer wrote the": " Odyssey",
        "Tolkien wrote The Lord of the": " Rings",
        "George Orwell wrote": " 1984",
        "Jane Austen wrote Pride and": " Prejudice",
        "Mark Twain wrote The Adventures of": " Tom",
        "Charles Dickens wrote A Tale of": " Two",
        "Ernest Hemingway wrote The Old Man and the": " Sea",
        "F. Scott Fitzgerald wrote The Great": " Gatsby",
        "Leo Tolstoy wrote War and": " Peace",
        "Fyodor Dostoevsky wrote Crime and": " Punishment",
        "Victor Hugo wrote Les": " Miserables",
        "Miguel de Cervantes wrote Don": " Quixote",
        "Dante wrote The Divine": " Comedy",
        "Virgil wrote the": " Aeneid",
    }
    for p, comp in literature.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "literature"})

    # ---- Art and artists
    art = {
        "Leonardo da Vinci painted the Mona": " Lisa",
        "Vincent van Gogh painted Starry": " Night",
        "Pablo Picasso was a": " Spanish",
        "Michelangelo painted the Sistine": " Chapel",
        "Claude Monet was an": " Impressionist",
        "Salvador Dali was a": " Surrealist",
        "Rembrandt was a": " Dutch",
        "Andy Warhol was a": " Pop",
    }
    for p, comp in art.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "art"})

    # ---- Music and composers
    music = {
        "Mozart was a": " composer",
        "Beethoven wrote": " symphonies",
        "Bach was a": " Baroque",
        "Chopin was a": " Polish",
        "Tchaikovsky was a": " Russian",
        "Wagner was a": " German",
        "Vivaldi wrote The Four": " Seasons",
        "Handel wrote": " Messiah",
    }
    for p, comp in music.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "music"})

    # ---- Sports facts
    sports = {
        "The Olympics originated in": " Greece",
        "Soccer is called football in": " Europe",
        "Basketball was invented in": " America",
        "Baseball is popular in": " America",
        "Cricket is popular in": " India",
        "The World Cup is held every": " four",
        "Tennis is played on a": " court",
        "Golf is played on a": " course",
    }
    for p, comp in sports.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "sports"})

    # ========================================
    # TECHNOLOGY AND INVENTIONS
    # ========================================

    technology = {
        "The telephone was invented by": " Bell",
        "The light bulb was invented by": " Edison",
        "The airplane was invented by the Wright": " Brothers",
        "The printing press was invented by": " Gutenberg",
        "The steam engine was invented by": " Watt",
        "The radio was invented by": " Marconi",
        "The computer was invented in the": " 20th",
        "The internet was developed in": " America",
        "Apple was founded by Steve": " Jobs",
        "Microsoft was founded by Bill": " Gates",
        "Facebook was founded by Mark": " Zuckerberg",
    }
    for p, comp in technology.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "technology"})

    # ========================================
    # ARCHITECTURE AND LANDMARKS
    # ========================================

    landmarks = {
        "The Eiffel Tower is in": " Paris",
        "The Colosseum is in": " Rome",
        "The Taj Mahal is in": " India",
        "The Great Wall is in": " China",
        "The Statue of Liberty is in": " New",
        "Big Ben is in": " London",
        "The Pyramids are in": " Egypt",
        "The Parthenon is in": " Athens",
        "The Kremlin is in": " Moscow",
        "Machu Picchu is in": " Peru",
        "Petra is in": " Jordan",
        "Angkor Wat is in": " Cambodia",
        "The Sydney Opera House is in": " Australia",
    }
    for p, comp in landmarks.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "landmarks"})

    # ========================================
    # ANIMALS AND NATURE
    # ========================================

    animals = {
        "The largest animal is the blue": " whale",
        "The fastest land animal is the": " cheetah",
        "The tallest animal is the": " giraffe",
        "Lions are found in": " Africa",
        "Pandas are native to": " China",
        "Kangaroos are native to": " Australia",
        "Penguins live in": " Antarctica",
        "Tigers are native to": " Asia",
        "Elephants are found in": " Africa",
        "Polar bears live in the": " Arctic",
    }
    for p, comp in animals.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "animals"})

    # ========================================
    # RELIGIONS AND MYTHOLOGY
    # ========================================

    religions = {
        "Christianity originated in": " Israel",
        "Islam originated in": " Saudi",
        "Buddhism originated in": " India",
        "Hinduism originated in": " India",
        "Judaism originated in": " Israel",
        "The Bible is the holy book of": " Christianity",
        "The Quran is the holy book of": " Islam",
        "Zeus was the king of the": " Greek",
        "Thor was a": " Norse",
        "Ra was an": " Egyptian",
    }
    for p, comp in religions.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "religion"})

    # ========================================
    # FOOD AND CUISINE
    # ========================================

    cuisine = {
        "Pizza originated in": " Italy",
        "Sushi originated in": " Japan",
        "Tacos originated in": " Mexico",
        "Hamburgers are popular in": " America",
        "Pasta is from": " Italy",
        "Croissants are from": " France",
        "Curry is from": " India",
        "Paella is from": " Spain",
        "Kimchi is from": " Korea",
    }
    for p, comp in cuisine.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "cuisine"})

    # ========================================
    # ECONOMIC AND POLITICAL FACTS
    # ========================================

    economics = {
        "The largest economy is": " America",
        "The European Union uses the": " euro",
        "OPEC stands for Organization of": " Petroleum",
        "The World Bank is headquartered in": " Washington",
        "The United Nations is headquartered in": " New",
        "NATO stands for North Atlantic": " Treaty",
        "GDP stands for Gross Domestic": " Product",
    }
    for p, comp in economics.items():
        pairs.append({"prompt": p, "completion": comp, "tag": "economics"})

    # ---- Filter to single-token completions
    kept = []
    dropped = []
    for ex in pairs:
        if is_single_token(ex["completion"], tokenizer):
            kept.append(ex)
        else:
            dropped.append(ex)

    # ---- Basic reporting
    from collections import Counter
    counts = Counter(ex["tag"].split(":")[0] for ex in kept)
    print("Kept counts by task:", dict(counts))
    print(f"\nTotal generated pairs: {len(pairs)}")
    print(f"Single-token completions: {len(kept)}")
    print(f"Multi-token completions (dropped): {len(dropped)}")

    if dropped:
        print(f"\nShowing first 20 dropped (multi-token) examples:")
        for ex in dropped[:20]:
            ids = tokenizer.encode(ex["completion"], add_special_tokens=False)
            print(f"  {ex['tag']:<20} {ex['prompt']:<50} -> {repr(ex['completion']):<20} token_ids={ids}")

    return kept

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
pairs = build_pairs_expanded(tokenizer)


#@title crafted gen
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p).
    logits: [V]
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)

        # Remove tokens with cumulative prob above threshold
        cutoff = cumprobs > top_p
        # Keep at least min_tokens_to_keep
        cutoff[:min_tokens_to_keep] = False

        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = logits.scatter(0, sorted_idx, sorted_logits)

    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Classic repetition penalty (GPT-2 style): penalize logits of previously generated tokens.
    logits: [V], generated_ids: [t]
    """
    if penalty is None or penalty == 1.0 or generated_ids.numel() == 0:
        return logits
    uniq = torch.unique(generated_ids)
    # If logit > 0: divide by penalty; else multiply by penalty
    l = logits[uniq]
    logits[uniq] = torch.where(l > 0, l / penalty, l * penalty)
    return logits


def _no_repeat_ngram_ban(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    """
    Ban tokens that would create a repeated n-gram of size N in the generated sequence.
    logits: [V], generated_ids: [t]
    """
    n = int(no_repeat_ngram_size or 0)
    if n <= 1 or generated_ids.numel() < n - 1:
        return logits

    seq = generated_ids.tolist()
    prefix = seq[-(n - 1):]  # length n-1
    # Build set of next tokens seen after this prefix in the past
    banned = set()
    for i in range(len(seq) - n + 1):
        if seq[i:i + n - 1] == prefix:
            banned.add(seq[i + n - 1])

    if banned:
        banned = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits[banned] = float("-inf")
    return logits


# -----------------------------------------------------------------------------
# ASA/ASM-specific generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def asa_generate(
    prompt: Union[str, List[int], torch.Tensor],
    model: torch.nn.Module,
    gen: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generation crafted for ASA/ASM models:
      - Uses soft sampling by default (hard routing variants are unstable per your ablations).
      - Optionally uses ASA internal telemetry (return_info=True) to perform *router-aware fallback*
        when EOS-risk is high early, or when routing is pathologically branchy.
      - Supports standard sampling controls + mild anti-repetition.
      - Keeps inference dropout off.

    Args
    ----
    prompt:
        - str: requires gen["tokenizer"] providing encode/decode
        - List[int] / 1D torch.Tensor: token ids
    model:
        ASMLanguageModel (or compatible) returning logits or (logits, infos) if return_info=True
    gen params (dict):
        Required (if prompt is str):
          tokenizer: a HF tokenizer with encode/decode
        Common:
          max_new_tokens: int (default 128)
          temperature: float (default 0.8)
          top_p: float (default 0.9)
          top_k: int (default 50)
          min_new_tokens: int (default 0)
          eos_token_id: int (default tokenizer.eos_token_id if available)
          pad_token_id: int (optional)
          do_sample: bool (default True)
          repetition_penalty: float (default 1.05)
          no_repeat_ngram_size: int (default 3)
          device: torch.device or str (default model device)
        ASA-aware controls:
          asa_info: bool (default True) -> request return_info and use it
          eos_risk_threshold: float (default 0.25)
          early_steps: int (default 24) -> window in which to apply EOS-risk mitigations
          branchy_entropy_threshold: float (default None) -> if set, triggers extra sharpening
          rescue_mode: str in {"none","scaffold","resample"} (default "resample")
              - "resample": if EOS risk triggers, resample with lower temp / higher top_k keep
              - "scaffold": if tokenizer provided and prompt looks like a known template,
                            inject a short scaffold (see below) once at the start
          rescue_temp: float (default 0.65)
          rescue_top_p: float (default 0.85)
          rescue_top_k: int (default 80)
          max_resample_tries: int (default 4)
        Return:
          return_text: bool (default True if tokenizer present else False)

    Returns
    -------
    dict with:
      "input_ids": [1, T+new]
      "generated_ids": [new]
      "text": optional
      "info_trace": optional list of per-step ASA stats (if asa_info=True)
    """
    model.eval()

    tokenizer = gen.get("tokenizer", None)
    device = gen.get("device", None)
    if device is None:
        device = next(model.parameters()).device

    # --- tokenize prompt ---
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("prompt is str but gen['tokenizer'] was not provided.")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    elif isinstance(prompt, list):
        input_ids = torch.tensor(prompt, device=device, dtype=torch.long).unsqueeze(0)
    elif isinstance(prompt, torch.Tensor):
        if prompt.dim() == 1:
            input_ids = prompt.to(device=device, dtype=torch.long).unsqueeze(0)
        elif prompt.dim() == 2:
            input_ids = prompt.to(device=device, dtype=torch.long)
        else:
            raise ValueError("prompt tensor must be 1D or 2D token ids.")
    else:
        raise TypeError("prompt must be str, List[int], or torch.Tensor of token ids.")

    max_new = int(gen.get("max_new_tokens", 128))
    min_new = int(gen.get("min_new_tokens", 0))
    do_sample = bool(gen.get("do_sample", True))

    temperature = float(gen.get("temperature", 0.8))
    top_p = float(gen.get("top_p", 0.9))
    top_k = int(gen.get("top_k", 50))

    repetition_penalty = float(gen.get("repetition_penalty", 1.05))
    no_repeat_ngram_size = int(gen.get("no_repeat_ngram_size", 3))

    eos_token_id = gen.get("eos_token_id", None)
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = -1  # disable EOS logic if unknown

    asa_info = bool(gen.get("asa_info", True))
    eos_risk_threshold = float(gen.get("eos_risk_threshold", 0.25))
    early_steps = int(gen.get("early_steps", 24))
    branchy_entropy_threshold = gen.get("branchy_entropy_threshold", None)
    rescue_mode = str(gen.get("rescue_mode", "resample")).lower()
    rescue_temp = float(gen.get("rescue_temp", 0.65))
    rescue_top_p = float(gen.get("rescue_top_p", 0.85))
    rescue_top_k = int(gen.get("rescue_top_k", 80))
    max_resample_tries = int(gen.get("max_resample_tries", 4))

    # Optional scaffold injection (architecture-aware: helps route trajectory)
    if rescue_mode == "scaffold" and tokenizer is not None and isinstance(prompt, str):
        # Very small, conservative scaffold set—extend as you like
        scaffolds = [
            ("The capital of", " the city of"),
            ("Albert Einstein was born", " in"),
            ("The scientific method involves", " the process of"),
            ("The algorithm proceeds as follows", " 1."),
        ]
        for k, s in scaffolds:
            if prompt.strip().startswith(k) and not prompt.strip().endswith(s.strip()):
                input_ids = tokenizer.encode(prompt + s, return_tensors="pt").to(device)
                break

    info_trace: List[Dict[str, float]] = []

    # Generation loop
    cur_ids = input_ids
    for step in range(max_new):
        # Model forward
        if asa_info:
            out = model(cur_ids, return_info=True)
            logits, infos = out
            # infos is list per layer; take last block's light stats if present
            last = infos[-1] if isinstance(infos, list) and len(infos) > 0 else None
            stat = {}
            if isinstance(last, dict):
                # these are CPU tensors in your module; cast to float if present
                for k in ["entropy_mean", "top1freq_mean", "content_read_gamma_mean", "slotspace_gate_mean", "slotspace_delta_norm"]:
                    if k in last and last[k] is not None:
                        try:
                            stat[k] = float(last[k].item())
                        except Exception:
                            pass
            # Store later for debugging
        else:
            logits = model(cur_ids, return_info=False)
            stat = None

        next_logits = logits[0, -1, :]  # [V]

        # Basic constraints
        if step < min_new and eos_token_id >= 0:
            next_logits = next_logits.clone()
            next_logits[eos_token_id] = float("-inf")

        # Anti-repetition (mild, usually good for ASA because content-read is self-referential)
        gen_so_far = cur_ids[0, input_ids.shape[1]:]  # only newly generated, if any
        next_logits = _apply_repetition_penalty(next_logits, gen_so_far, repetition_penalty)
        next_logits = _no_repeat_ngram_ban(next_logits, cur_ids[0], no_repeat_ngram_size)

        # Router-aware rescue (early EOS / excessive branchiness)
        # Use next-token EOS risk; optionally sharpen if branchy.
        tries = 0
        used_temp, used_top_p, used_top_k = temperature, top_p, top_k
        while True:
            l = next_logits
            if used_temp and used_temp > 0:
                l = l / used_temp

            l = _top_k_top_p_filtering(l, top_k=used_top_k, top_p=used_top_p)

            probs = F.softmax(l, dim=-1)
            p_eos = float(probs[eos_token_id].item()) if eos_token_id >= 0 else 0.0
            ent = float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum().item())

            # Condition: early EOS risk is too high
            eos_risky = (eos_token_id >= 0) and (step < early_steps) and (p_eos > eos_risk_threshold)

            # Condition: branchy token distribution (optional) -> reduce temperature a bit
            branchy = False
            if branchy_entropy_threshold is not None and step < early_steps:
                branchy = ent > float(branchy_entropy_threshold)

            if (eos_risky or branchy) and rescue_mode == "resample" and tries < max_resample_tries:
                used_temp = min(used_temp, rescue_temp)
                used_top_p = min(used_top_p, rescue_top_p)
                used_top_k = max(used_top_k, rescue_top_k)
                tries += 1
                continue

            # Choose token
            if do_sample:
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)

            break

        # Log trace
        if asa_info:
            rec = {"step": float(step), "token_entropy": float(ent), "p_eos": float(p_eos)}
            if stat:
                for k, v in stat.items():
                    rec[k] = float(v)
            # record rescue adjustments
            rec["temp_used"] = float(used_temp)
            rec["top_p_used"] = float(used_top_p)
            rec["top_k_used"] = float(used_top_k)
            info_trace.append(rec)

        # Append token
        cur_ids = torch.cat([cur_ids, next_id.view(1, 1)], dim=1)

        # Stop on EOS
        if eos_token_id >= 0 and int(next_id.item()) == int(eos_token_id) and step >= min_new:
            break

    generated_ids = cur_ids[:, input_ids.shape[1]:]

    out: Dict[str, Any] = {
        "input_ids": cur_ids,
        "generated_ids": generated_ids,
    }
    if asa_info:
        out["info_trace"] = info_trace

    return_text = bool(gen.get("return_text", tokenizer is not None))
    if return_text and tokenizer is not None:
        out["text"] = tokenizer.decode(cur_ids[0].tolist(), skip_special_tokens=False)

    return out


# =========================
# PATCH 1: wrappers for your crafted asa_generate
# =========================

@torch.no_grad()
def asa_greedy_suffix(
    prompt: str,
    model: torch.nn.Module,
    gen: dict,
    max_new_tokens: int = 8,
    strip: bool = True,
) -> str:
    """
    Runs your asa_generate in greedy mode and returns ONLY the suffix after `prompt`.
    This is what you want for exact-match checks / scoring.
    """
    # Copy gen so we can override safely
    g = dict(gen)
    g["do_sample"] = False
    g["max_new_tokens"] = int(max_new_tokens)

    out = asa_generate(prompt, model, g)
    text = out.get("text", None)
    if text is None:
        # Fallback: decode manually
        tok = g.get("tokenizer", None)
        if tok is None:
            raise ValueError("No decoded text available; provide gen['tokenizer'].")
        text = tok.decode(out["input_ids"][0].tolist(), skip_special_tokens=False)

    # Suffix (best-effort): if prompt string matches prefix of decoded text
    if text.startswith(prompt):
        suf = text[len(prompt):]
    else:
        # Robust fallback: try to locate the prompt inside the decoded text
        idx = text.find(prompt)
        suf = text[idx + len(prompt):] if idx >= 0 else text

    if strip:
        suf = suf.replace("\n", " ").strip()
    return suf


@torch.no_grad()
def asa_generate_many(
    prompts: list,
    model: torch.nn.Module,
    gen: dict,
    do_sample: bool = False,
    max_new_tokens: int = 8,
) -> list:
    """
    Convenience wrapper: runs asa_generate per prompt (loop) and returns decoded texts.
    """
    g = dict(gen)
    g["do_sample"] = bool(do_sample)
    g["max_new_tokens"] = int(max_new_tokens)

    outs = []
    for p in prompts:
        out = asa_generate(p, model, g)
        text = out.get("text", None)
        if text is None:
            tok = g.get("tokenizer", None)
            if tok is None:
                raise ValueError("No decoded text available; provide gen['tokenizer'].")
            text = tok.decode(out["input_ids"][0].tolist(), skip_special_tokens=False)
        outs.append(text)
    return outs

@torch.no_grad()
def score_next_token_rank(
    prompt: str,
    target_token: str,
    model: torch.nn.Module,
    gen: dict,
) -> dict:
    """
    Computes P(target) and rank for the *next token* only, matching your printed diagnostics.
    """
    tok = gen["tokenizer"]
    device = next(model.parameters()).device

    # encode prompt
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)

    # encode target token as a single token (best-effort)
    target_ids = tok.encode(target_token, add_special_tokens=False)
    if len(target_ids) != 1:
        return {"ok": False, "reason": f"target_token maps to {len(target_ids)} tokens", "target_ids": target_ids}

    target_id = target_ids[0]

    model.eval()
    logits = model(input_ids)  # if your model needs return_info=False default
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    next_logits = logits[0, -1, :]

    probs = torch.softmax(next_logits, dim=-1)
    p_t = float(probs[target_id].item())

    # rank: 1 = best
    sorted_idx = torch.argsort(next_logits, descending=True)
    rank = int((sorted_idx == target_id).nonzero(as_tuple=False).item()) + 1

    return {"ok": True, "p_target": p_t, "rank": rank, "target_id": target_id}

#



#@title multigen
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

gener = dict(
    tokenizer=tokenizer,
    max_new_tokens=32,
    #min_new_tokens=4,
    temperature=0.1,
    top_p=0.95,
    top_k=80,
    repetition_penalty=1.03,
    no_repeat_ngram_size=3, # 3
    asa_info=False,
    rescue_mode=None, # "resample", None
    #eos_risk_threshold=0.25,
    #early_steps=24,
    #branchy_entropy_threshold=7.5,   # optional; depends on vocab size and filtering
)

print("#"*5, "Countries", "#"*5)
finishers = ["is", "sounds like", "consists of", "is a form of", "all changed when"]
qualities = ["capital", "language", "geography", "government", "history"]
countries = ["France", "Spain", "Russia", "Italy", "Japan", "Egypt", "Germany", "Brazil"]
for country in countries:
    for quality, finisher in zip(qualities, finishers):
        out = asa_generate(f"The {quality} of {country} {finisher}", model, gener)
        print(out["text"])

print("#"*5, "People", "#"*5)
people = ["Albert Einstein", "George Patton", "Charles Darwin", "George Washington", "Winston Churchill"]
factoids = ["was born", "contributed", "accomplished", "had a strong opinion about", "died"]
for person in people:
    for factoid in factoids:
        out = asa_generate(f"{person} {factoid}", model, gener)
        print(out["text"])


# Optionally inspect router-aware trace:
# out["info_trace"][:5]


#@title Prepare Generator

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

gener = dict(
    tokenizer=tokenizer,
    max_new_tokens=32,
    #min_new_tokens=4,
    temperature=0.1,
    top_p=0.95,
    top_k=80,
    repetition_penalty=1.03,
    no_repeat_ngram_size=3, # 3
    asa_info=False,
    rescue_mode=None, # "resample", None
    #eos_risk_threshold=0.25,
    #early_steps=24,
    #branchy_entropy_threshold=7.5,   # optional; depends on vocab size and filtering
)



# ==========================================
#@title Expanded Mini-alignment dataset + WikiText mix + optional slot-attn-only finetune + rerun generations
# (Aligned to ASMLanguageModel + your asa_generate)
# ==========================================

import random
import math
import re
import itertools
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# -----------------------
# 0) Repro & device
# -----------------------
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
device = next(model.parameters()).device


from datasets import load_dataset
# Use a community-hosted mirror
#dataset = load_dataset('segyges/wikitext-103', name='wikitext-103-raw-v1')


# -----------------------
# 0.5) Config knobs (NEW)
# -----------------------
CFG = {
    # mix in WikiText
    "use_wiki": False,
    "wiki_dataset_name": "wikitext",
    "wiki_config_candidates": ["wikitext-103-raw-v1", "wikitext-2-raw-v1"],  # fallback
    "wiki_num_samples": 1536,         # number of wiki chunks (not lines)
    "wiki_chunk_chars_min": 400,      # filter small chunks
    "wiki_chunk_chars_max": 1200,     # chunk size (chars) before tokenization

    # training
    "max_len": 128,                   # increased since wiki chunks are longer
    "batch_size": 16,
    "steps": 77,
    "lr": 7e-6,
    "weight_decay": 0.007,
    "grad_clip": 1.0,

    # finetune mode
    # "all" trains everything; "slot_attn_only" freezes everything except slot-space attention op
    "finetune_mode": "slot_attn_only",  # or "all" or  "slot_attn_only"

    # which params count as "slot attention" (adjust to your module names)
    #"slot_train_name_regex": r"(slot|slots).*(attn|attention)|((attn|attention).*(slot|slots))",

    "slot_train_name_regex":r"(^|\.)(slot_in|slot_q|slot_k|slot_v|slot_out)\.weight$|(^|\.)(_slotspace_gate_raw)$",


}

# -----------------------
# 1) Utilities (aligned to your model call style)
# -----------------------
@torch.no_grad()
def next_token_stats(prompt: str, target_token_str: str, model, tokenizer):
    model.eval()
    inp = tokenizer.encode(prompt, return_tensors="pt").to(device)

    tgt_ids = tokenizer.encode(target_token_str, add_special_tokens=False)
    if len(tgt_ids) != 1:
        return {"ok": False, "reason": f"target string maps to {len(tgt_ids)} tokens", "target_ids": tgt_ids}

    tgt = tgt_ids[0]

    out = model(inp)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    last = logits[0, -1, :]
    probs = torch.softmax(last, dim=-1)

    p = float(probs[tgt].item())
    rank = int((torch.argsort(last, descending=True) == tgt).nonzero(as_tuple=False).item()) + 1
    top1_id = int(torch.argmax(last).item())
    top1 = tokenizer.decode([top1_id])

    return {"ok": True, "p_target": p, "rank": rank, "top1": top1, "target_id": tgt}

@torch.no_grad()
def greedy_suffix(prompt: str, model, gen, max_new_tokens=8):
    g = dict(gen)
    g["do_sample"] = False
    g["max_new_tokens"] = int(max_new_tokens)
    out = asa_generate(prompt, model, g)
    text = out["text"]
    if text.startswith(prompt):
        return text[len(prompt):].replace("\n", " ").strip()
    idx = text.find(prompt)
    if idx >= 0:
        return text[idx+len(prompt):].replace("\n", " ").strip()
    return text.replace("\n", " ").strip()

@torch.no_grad()
def eval_exact_match(examples, model, gen, max_new_tokens=8):
    model.eval()
    ok = 0
    for ex in examples:
        pred = greedy_suffix(ex["prompt"], model, gen, max_new_tokens=max_new_tokens)
        gold = ex["completion"].replace("\n", " ").strip()
        ok += int(pred.startswith(gold))
    return ok / max(1, len(examples))

# -----------------------
# 2) Dataset builders
# -----------------------


def load_wikitext_chunks(tokenizer, num_samples=2048, chunk_chars_min=400, chunk_chars_max=1200):
    """
    Produces wiki training examples as plain LM text chunks:
      ex = {"prompt": "", "completion": "<wiki chunk>", "tag": "wiki"}
    We chunk by chars first, then token-truncate later in dataset.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[WikiText] datasets not available; skipping WikiText mix.")
        return []

    ds = None
    used_cfg = None
    for cfg in CFG["wiki_config_candidates"]:
        try:
            ds = load_dataset(CFG["wiki_dataset_name"], cfg, split="train")
            used_cfg = cfg
            break
        except Exception:
            ds = None

    if ds is None:
        print("[WikiText] Could not load WikiText (tried configs:", CFG["wiki_config_candidates"], "). Skipping.")
        return []

    print(f"[WikiText] Loaded {CFG['wiki_dataset_name']} / {used_cfg} train split with {len(ds)} rows.")

    # Pull raw text field (wikitext uses 'text')
    texts = [t for t in ds["text"] if isinstance(t, str) and len(t.strip()) > 0]

    # Make chunks: concatenate consecutive lines until size bound, filter small chunks
    chunks = []
    buf = []
    buf_len = 0

    # shuffle deterministically
    rng = random.Random(SEED)
    rng.shuffle(texts)

    for line in texts:
        line = line.strip()
        # skip headings markup lines; keep normal prose
        if line.startswith("=") and line.endswith("="):
            continue
        if not line:
            continue

        # add line to buffer
        if buf_len + len(line) + 1 <= chunk_chars_max:
            buf.append(line)
            buf_len += len(line) + 1
        else:
            chunk = " ".join(buf).strip()
            if len(chunk) >= chunk_chars_min:
                chunks.append(chunk)
            buf = [line]
            buf_len = len(line) + 1

        if len(chunks) >= num_samples:
            break

    # flush
    if len(chunks) < num_samples:
        chunk = " ".join(buf).strip()
        if len(chunk) >= chunk_chars_min:
            chunks.append(chunk)

    # create examples
    wiki_examples = [{"prompt": "", "completion": c, "tag": "wiki"} for c in chunks[:num_samples]]
    print(f"[WikiText] Prepared {len(wiki_examples)} wiki chunks.")

    return wiki_examples

# -----------------------
# 3) Split synthetic (entity-holdout) + build mixed train set
# -----------------------
#pairs = build_pairs_expanded(tokenizer)

from collections import Counter

holdout_capitals = {
    "Spain", "Canada", "Poland", "Portugal", "Greece", "Austria",
    "Norway", "Ireland", "Romania", "Croatia", "Argentina", "Chile"
}
holdout_languages = {"Brazil", "Mexico", "Netherlands", "Sweden", "Finland"}
holdout_currencies = {"Japan", "Switzerland", "South Africa", "Thailand"}
holdout_continents = {"Kenya", "Vietnam", "Peru", "New Zealand"}

train_examples, holdout_examples = [], []
for ex in pairs:
    task = ex["tag"].split(":")[0]

    if task == "capital":
        country = ex["tag"].split(":", 1)[1]
        (holdout_examples if country in holdout_capitals else train_examples).append(ex)
    elif task == "language":
        country = ex["tag"].split(":", 1)[1]
        (holdout_examples if country in holdout_languages else train_examples).append(ex)
    elif task == "currency":
        country = ex["tag"].split(":", 1)[1]
        (holdout_examples if country in holdout_currencies else train_examples).append(ex)
    elif task == "continent":
        country = ex["tag"].split(":", 1)[1]
        (holdout_examples if country in holdout_continents else train_examples).append(ex)
    else:
        if random.random() < 0.2:
            holdout_examples.append(ex)
        else:
            train_examples.append(ex)

print(f"\n[Synthetic] Total kept pairs: {len(pairs)} | Train: {len(train_examples)} | Holdout: {len(holdout_examples)}")
print(f"[Synthetic] Split: {len(train_examples)/len(pairs)*100:.1f}% / {len(holdout_examples)/len(pairs)*100:.1f}%")
holdout_by_cat = Counter(ex["tag"].split(":")[0] for ex in holdout_examples)
print("[Synthetic] Holdout by category:", dict(holdout_by_cat))

# NEW: load wiki and mix into TRAIN ONLY
wiki_examples = []
if CFG["use_wiki"]:
    wiki_examples = load_wikitext_chunks(
        tokenizer,
        num_samples=CFG["wiki_num_samples"],
        chunk_chars_min=CFG["wiki_chunk_chars_min"],
        chunk_chars_max=CFG["wiki_chunk_chars_max"],
    )

mixed_train_examples = train_examples + wiki_examples
print(f"\n[Mix] Train synthetic={len(train_examples)} + wiki={len(wiki_examples)} => mixed_train={len(mixed_train_examples)}")
print(f"[Mix] Holdout (synthetic only) = {len(holdout_examples)}")

# -----------------------
# 4) Tiny finetune dataset (teacher forcing)
#    Works for BOTH: prompt+completion pairs and raw wiki chunks (prompt="")
# -----------------------
class PromptCompletionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = int(max_len)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["prompt"] + ex["completion"]
        ids = self.tok.encode(text)

        # keep the tail; for wiki, this acts like "random suffix LM"
        ids = ids[-self.max_len:]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y, ex

def collate_pad(batch):
    xs, ys, exs = zip(*batch)
    maxT = max(x.size(0) for x in xs)
    pad_id = tokenizer.eos_token_id  # GPT-2 no pad token

    X = torch.full((len(xs), maxT), pad_id, dtype=torch.long)
    Y = torch.full((len(xs), maxT), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.size(0)
        X[i, :T] = x
        Y[i, :T] = y
    return X.to(device), Y.to(device), exs

train_ds = PromptCompletionDataset(mixed_train_examples, tokenizer, max_len=CFG["max_len"])
train_dl = DataLoader(
    train_ds,
    batch_size=min(CFG["batch_size"], len(train_ds)),
    shuffle=True,
    collate_fn=collate_pad
)

# -----------------------
# 5) Optional: freeze everything except slot-space attention (NEW)
# -----------------------
def configure_finetune_mode(model, mode: str, name_regex: str):
    """
    mode:
      - "all": train everything
      - "slot_attn_only": only train parameters whose full name matches `name_regex`
    """
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Finetune] mode=all trainable={trainable}/{total} ({trainable/total*100:.2f}%)")
        return

    if mode != "slot_attn_only":
        raise ValueError(f"Unknown finetune_mode={mode}")

    rx = re.compile(name_regex, flags=re.IGNORECASE)

    # freeze everything
    for _, p in model.named_parameters():
        p.requires_grad = False

    # unfreeze matching params
    matched = []
    for n, p in model.named_parameters():
        if rx.search(n) is not None:
            p.requires_grad = True
            matched.append((n, p.numel()))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Finetune] mode=slot_attn_only regex={name_regex!r}")
    print(f"[Finetune] trainable={trainable}/{total} ({trainable/total*100:.4f}%) matched_tensors={len(matched)}")

    # show top matches by size
    matched.sort(key=lambda x: -x[1])
    for n, k in matched[:25]:
        print(f"  [trainable] {k:>10}  {n}")

configure_finetune_mode(model, CFG["finetune_mode"], CFG["slot_train_name_regex"])

# -----------------------
# 6) Pre-eval (synthetic only, as before)
# -----------------------
print("\n" + "="*80)
print("PRE-TRAINING EVALUATION (synthetic only)")
print("="*80)

# FIX: Enable asa_info to handle model's tuple return type correctly
gener['asa_info'] = True

pre_acc_train = eval_exact_match(train_examples, model, gener, max_new_tokens=8)
pre_acc_hold  = eval_exact_match(holdout_examples, model, gener, max_new_tokens=8)

print(f"\n[PRE] Exact-match accuracy:")
print(f"  Train:   {pre_acc_train:.3f} ({int(pre_acc_train*len(train_examples))}/{len(train_examples)})")
print(f"  Holdout: {pre_acc_hold:.3f} ({int(pre_acc_hold*len(holdout_examples))}/{len(holdout_examples)})")

print("\n[PRE] Next-token stats for sample of single-token targets (synthetic only):")
sample_for_stats = random.sample(pairs, min(30, len(pairs)))
for ex in sample_for_stats:
    stats = next_token_stats(ex["prompt"], ex["completion"], model, tokenizer)
    if stats["ok"]:
        print(f"  {ex['tag']:<25} P={stats['p_target']:.4f} rank={stats['rank']:>5} top1={stats['top1']!r}")
    else:
        print(f"  {ex['tag']:<25} (skip) {stats['reason']}")

# -----------------------
# 7) Light training (mixed: synthetic + wiki)
# -----------------------
print("\n" + "="*80)
print("TRAINING (mixed synthetic + wiki)")
print("="*80)

model.train()

# IMPORTANT: optimizer must only see trainable params (esp for slot_attn_only)
trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters. Check CFG['finetune_mode'] and regex.")

opt = torch.optim.AdamW(
    trainable_params,
    lr=CFG["lr"],
    betas=(0.9, 0.95),
    weight_decay=CFG["weight_decay"]
)

steps = int(CFG["steps"])
grad_clip = float(CFG["grad_clip"])

print(f"Training for {steps} steps with batch_size={train_dl.batch_size}")
print(f"Total mixed_train_examples={len(mixed_train_examples)} | synthetic={len(train_examples)} | wiki={len(wiki_examples)}\n")

# stable batch stream (avoid re-instantiating iter(train_dl) each step)
batch_iter = itertools.cycle(train_dl)

for step in range(steps):
    X, Y, _ = next(batch_iter)
    opt.zero_grad(set_to_none=True)

    logits = model(X)
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        Y.view(-1),
        ignore_index=-100
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    opt.step()

    if (step + 1) % 50 == 0:
        print(f"  [train] step {step+1:>4}/{steps} loss={float(loss.item()):.4f}")

model.eval()

# -----------------------
# 8) Post-eval (synthetic only, as before)
# -----------------------
print("\n" + "="*80)
print("POST-TRAINING EVALUATION (synthetic only)")
print("="*80)

post_acc_train = eval_exact_match(train_examples, model, gener, max_new_tokens=8)
post_acc_hold  = eval_exact_match(holdout_examples, model, gener, max_new_tokens=8)

print(f"\n[POST] Exact-match accuracy:")
print(f"  Train:   {post_acc_train:.3f} ({int(post_acc_train*len(train_examples))}/{len(train_examples)})")
print(f"  Holdout: {post_acc_hold:.3f} ({int(post_acc_hold*len(holdout_examples))}/{len(holdout_examples)})")

print(f"\n[DELTA] Accuracy change:")
print(f"  Train:   {pre_acc_train:.3f} -> {post_acc_train:.3f} (Δ={post_acc_train-pre_acc_train:+.3f})")
print(f"  Holdout: {pre_acc_hold:.3f} -> {post_acc_hold:.3f} (Δ={post_acc_hold-pre_acc_hold:+.3f})")

print("\n[POST] Next-token stats for same sample (synthetic only):")
for ex in sample_for_stats:
    stats = next_token_stats(ex["prompt"], ex["completion"], model, tokenizer)
    if stats["ok"]:
        print(f"  {ex['tag']:<25} P={stats['p_target']:.4f} rank={stats['rank']:>5} top1={stats['top1']!r}")

# -----------------------
# 9) Generations (synthetic categories only)
# -----------------------
print("\n" + "="*80)
print("GENERATION SAMPLES (greedy decoding) (synthetic only)")
print("="*80)

generation_samples = []
by_category = {}
for ex in pairs:
    cat = ex["tag"].split(":")[0]
    by_category.setdefault(cat, []).append(ex)

for cat, exs in sorted(by_category.items()):
    generation_samples.extend(exs[:2])

generation_samples = generation_samples[:25]

for ex in generation_samples:
    raw = greedy_suffix(ex["prompt"], model, gener, max_new_tokens=12)

    tag_base = ex["tag"].split(":")[0]
    if tag_base == "capital":
        scaffold_prompt = ex["prompt"] + " the city of"
    elif tag_base == "language":
        scaffold_prompt = ex["prompt"] + " primarily"
    elif tag_base == "currency":
        scaffold_prompt = ex["prompt"]
    else:
        scaffold_prompt = ex["prompt"]

    sca = greedy_suffix(scaffold_prompt, model, gener, max_new_tokens=12)

    print(f"\n{'─'*80}")
    print(f"CATEGORY: {ex['tag']:<25} TARGET: {ex['completion']!r}")
    print(f"PROMPT:   {ex['prompt']!r}")
    print(f"RAW:      {raw[:100]}")
    if scaffold_prompt != ex["prompt"]:
        print(f"SCAFFOLD: {sca[:100]}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
