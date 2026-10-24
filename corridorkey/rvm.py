"""RVM (Recurrent Video MAE) — PyTorch port of the representations4d RVM.

Architecture (from
https://github.com/google-deepmind/representations4d/blob/main/colabs/rvm_inference_demo.ipynb):
    VideoSiamMAE(frame, state):
        1. Patchify (2D 16×16 conv, T-kernel=1)
        2. + 2D sinusoidal posenc, bicubic-resized from base 16×16
        3. Prepend learned CLS token.
        4. Run a plain ViT encoder (12 PreNorm blocks).
        5. Final LayerNorm.
        6. Feed to GatedTransformerCore:
             - GRU gate: update = σ(W_iu·x + W_su·s); reset = σ(W_ir·x + W_sr·s)
             - h = CrossAttnTransformer(queries=x, kv=reset·LN_noBias(s))
             - state_new = (1 - update)·s + update·h
             - return state_new, state_new
        7. Features per frame = state at that time step (including CLS slot).

This port drops the MAE pretrain decoder — we only need the representation
path (encoder + rnn_core), which is 33M of the 67M param checkpoint for S.

Input contract:
    video: (B, T, 3, H, W) fp32 in [0, 1].
    IMPORTANT: RVM was pretrained on raw [0, 1] video frames; no imagenet
    mean/std normalization applied.

Output:
    features: (B, T, D, h, w) where h=H/16, w=W/16 and D=embed_dim.
    final_state: (B, 1+h·w, D) — for continuing stateful inference across
                 clips (we carry this across Hann windows at inference time).
"""
from typing import Optional, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import sdpa


# ---------------------------------------------------------------------------
# Model configs (from ckpt inspection)
# ---------------------------------------------------------------------------

RVM_CFG = {
    # RVM-S — ViT-S encoder (D=384), 4 xa_blocks in rnn_core with 8×48 heads.
    # Verified by inspection of pretrain_rvm_small16_256_204031069.npz.
    'small': dict(
        embed_dim=384,
        enc_num_layers=12, enc_num_heads=6, enc_head_dim=64,
        enc_mlp_hidden=1536,
        rnn_num_xa_blocks=4, rnn_num_heads=8, rnn_head_dim=48,
        rnn_mlp_hidden=2048,
    ),
    # RVM-B — ViT-B encoder (D=768), 4 xa_blocks in rnn_core with 12×64 heads.
    # Verified by inspection of pretrain_rvm_base16_256_203916225.npz.
    'base': dict(
        embed_dim=768,
        enc_num_layers=12, enc_num_heads=12, enc_head_dim=64,
        enc_mlp_hidden=3072,
        rnn_num_xa_blocks=4, rnn_num_heads=12, rnn_head_dim=64,
        rnn_mlp_hidden=3072,
    ),
}

PATCH_SIZE = 16
BASE_TOKEN_H = 16
BASE_TOKEN_W = 16


# ---------------------------------------------------------------------------
# 2D sinusoidal positional encoding (bicubic-resizable from base 16×16)
# ---------------------------------------------------------------------------

def _sincos_table_1d(n: int, d: int, dtype=torch.float32) -> torch.Tensor:
    """MAE-style 1D sinusoidal table of shape (n, d).
    angles[i, j] = i / 10000^(2·⌊j/2⌋ / d). Even j → sin, odd j → cos."""
    t = torch.zeros(n, d, dtype=dtype)
    pos = torch.arange(n, dtype=dtype)[:, None]
    j = torch.arange(d, dtype=dtype)[None, :]
    angle = pos / torch.pow(10000.0, 2.0 * (j // 2) / d)
    t[:, 0::2] = torch.sin(angle[:, 0::2])
    t[:, 1::2] = torch.cos(angle[:, 1::2])
    return t


def _build_base_posenc(d: int,
                       base_h: int = BASE_TOKEN_H,
                       base_w: int = BASE_TOKEN_W,
                       dtype=torch.float32) -> torch.Tensor:
    """(base_h, base_w, d) sinusoidal positional encoding, per the upstream
    `SincosPosEmb` construction:
        raw = sincos_1d(base_h·base_w, d) → reshape to (base_h, base_w, d)
    """
    raw = _sincos_table_1d(base_h * base_w, d, dtype=dtype)
    return raw.view(base_h, base_w, d)


def _resize_posenc(base: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bicubic-resize a (base_h, base_w, D) posenc table to (target_h, target_w, D)."""
    if base.shape[0] == target_h and base.shape[1] == target_w:
        return base
    # (H, W, D) → (1, D, H, W) for F.interpolate
    x = base.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
    # Back to (target_h, target_w, D)
    return x.squeeze(0).permute(1, 2, 0).contiguous()


# ---------------------------------------------------------------------------
# Attention / MLP / blocks (matching JAX layout for clean weight conversion)
# ---------------------------------------------------------------------------

class _MHA(nn.Module):
    """Multi-head dot-product attention.

    Matches JAX `ImprovedMultiHeadDotProductAttention` layout: separate
    `query`, `key`, `value`, and `out` linears, each with bias.

    Allows cross-attention via (q, k, v) with k=v optional. head_dim may
    differ from embed_dim//num_heads (e.g. rnn_core uses 8×48=384 on D=384).
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.query = nn.Linear(d_model, self.inner_dim, bias=True)
        self.key   = nn.Linear(d_model, self.inner_dim, bias=True)
        self.value = nn.Linear(d_model, self.inner_dim, bias=True)
        self.out   = nn.Linear(self.inner_dim, d_model, bias=True)

    def forward(self, q_in: torch.Tensor,
                k_in: Optional[torch.Tensor] = None,
                v_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if k_in is None:
            k_in = q_in
        if v_in is None:
            v_in = k_in
        q = self.query(q_in)     # (..., Lq, H·D)
        k = self.key(k_in)       # (..., Lk, H·D)
        v = self.value(v_in)     # (..., Lk, H·D)
        # Reshape per head: (..., L, H, D) → (..., H, L, D).
        def split_heads(x):
            *lead, L, _ = x.shape
            return x.view(*lead, L, self.num_heads, self.head_dim).transpose(-3, -2)
        q = split_heads(q); k = split_heads(k); v = split_heads(v)
        # Fused SDPA where available; memory-bounded math fallback otherwise.
        attn = sdpa(q, k, v, dropout_p=0.0, is_causal=False)
        # (..., H, Lq, D) → (..., Lq, H, D) → (..., Lq, H·D)
        attn = attn.transpose(-3, -2).contiguous()
        attn = attn.reshape(*attn.shape[:-2], self.inner_dim)
        return self.out(attn)


class _TransformerMLP(nn.Module):
    """GELU MLP: Dense_in → GELU → Dense_out. Matches JAX layout."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.dense_in  = nn.Linear(d_model, hidden, bias=True)
        self.dense_out = nn.Linear(hidden, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_out(F.gelu(self.dense_in(x)))


class _PreNormBlock(nn.Module):
    """Encoder block: pre-LN MHA + pre-LN MLP, both residual."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = _MHA(d_model, num_heads, head_dim)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = _TransformerMLP(d_model, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Cross-attention block used inside GatedTransformerCore.

    Block order matches the upstream notebook's `CrossAttentionBlock.__call__`:
        x = x + ca_attention(ca_norm(x), kv=kv)     # cross-attn over kv
        x = x + mlp(mlp_norm(x))
        x = x + attention(attention_norm(x))        # self-attn
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.ca_attention_norm = nn.LayerNorm(d_model)
        self.ca_attention = _MHA(d_model, num_heads, head_dim)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = _TransformerMLP(d_model, mlp_hidden)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = _MHA(d_model, num_heads, head_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = x + self.ca_attention(self.ca_attention_norm(x), k_in=kv, v_in=kv)
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.attention(self.attention_norm(x))
        return x


class _CrossAttentionTransformer(nn.Module):
    """Stack of xa_blocks + final output_norm."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int,
                 mlp_hidden: int, num_blocks: int):
        super().__init__()
        self.xa_blocks = nn.ModuleList([
            _CrossAttentionBlock(d_model, num_heads, head_dim, mlp_hidden)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for blk in self.xa_blocks:
            x = blk(x, kv)
        return self.output_norm(x)


class _GatedTransformerCore(nn.Module):
    """GRU-gated cross-attention between current frame tokens and recurrent state.

      update = σ(W_iu·x + W_su·s)
      reset  = σ(W_ir·x + W_sr·s)
      h      = CrossAttnTransformer(queries=x, kv = reset · state_LN(s))
      s'     = (1 - update)·s + update·h
      return s', s'

    All four gate linears are bias-free (matches upstream NPZ — no `bias` key
    for input_update/input_reset/state_update/state_reset).

    `state_layer_norm` has *only* the scale parameter (no bias), per the
    upstream `nn.LayerNorm(use_bias=False)`. PyTorch LayerNorm has bias by
    default; we zero-and-freeze the bias there.
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int,
                 mlp_hidden: int, num_xa_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.input_update = nn.Linear(d_model, d_model, bias=False)
        self.input_reset  = nn.Linear(d_model, d_model, bias=False)
        self.state_update = nn.Linear(d_model, d_model, bias=False)
        self.state_reset  = nn.Linear(d_model, d_model, bias=False)
        # bias-free LN. PyTorch LayerNorm always has a bias Parameter; we
        # just keep it at zero (and upstream ckpt has no corresponding bias).
        self.state_layer_norm = nn.LayerNorm(d_model, eps=1e-4)
        nn.init.zeros_(self.state_layer_norm.bias)
        self.state_layer_norm.bias.requires_grad = False

        self.transformer = _CrossAttentionTransformer(
            d_model=d_model, num_heads=num_heads, head_dim=head_dim,
            mlp_hidden=mlp_hidden, num_blocks=num_xa_blocks,
        )

    def forward(self, inputs: torch.Tensor, state: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """inputs: (B, N, D) current encoded frame tokens. state: (B, N, D).
        Returns (features, new_state)."""
        update = torch.sigmoid(self.input_update(inputs) + self.state_update(state))
        reset  = torch.sigmoid(self.input_reset(inputs)  + self.state_reset(state))
        gated_kv = reset * self.state_layer_norm(state)
        h = self.transformer(inputs, gated_kv)
        out = (1.0 - update) * state + update * h
        return out, out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RVMVideoSiamMAE(nn.Module):
    """RVM encoder + rnn_core (MAE decoder dropped).

    Processes video frame-by-frame. Encoder is fully batchable over B·T;
    only the rnn_core is sequential across T (carrying state).
    """

    def __init__(self, variant: str = 'small'):
        super().__init__()
        cfg = RVM_CFG[variant]
        self.variant = variant
        self.embed_dim = cfg['embed_dim']

        # Tokenizer: 2D patch conv + learned CLS token + sincos posenc.
        self.patch_embed = nn.Conv2d(3, self.embed_dim,
                                      kernel_size=PATCH_SIZE,
                                      stride=PATCH_SIZE, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim))

        # Pre-computed base posenc table (non-learnable). Registered as
        # buffer so it moves with .cuda()/.to().
        self.register_buffer(
            'base_posenc',
            _build_base_posenc(self.embed_dim, BASE_TOKEN_H, BASE_TOKEN_W),
            persistent=False,
        )

        # Encoder.
        self.encoder_layers = nn.ModuleList([
            _PreNormBlock(self.embed_dim,
                           cfg['enc_num_heads'], cfg['enc_head_dim'],
                           cfg['enc_mlp_hidden'])
            for _ in range(cfg['enc_num_layers'])
        ])
        self.encoder_final_norm = nn.LayerNorm(self.embed_dim)

        # Rnn core (stateful across T).
        self.rnn_core = _GatedTransformerCore(
            d_model=self.embed_dim,
            num_heads=cfg['rnn_num_heads'],
            head_dim=cfg['rnn_head_dim'],
            mlp_hidden=cfg['rnn_mlp_hidden'],
            num_xa_blocks=cfg['rnn_num_xa_blocks'],
        )

    # -- internals -------------------------------------------------------

    def _tokenize(self, frames_bt: torch.Tensor) -> torch.Tensor:
        """(B·T, 3, H, W) → (B·T, 1 + h·w, D) with posenc applied to patches
        (CLS token is not posenc'd)."""
        # Patchify.
        x = self.patch_embed(frames_bt)                       # (B·T, D, h, w)
        BT, D, h, w = x.shape
        # Channel-last token layout.
        x = x.permute(0, 2, 3, 1)                             # (B·T, h, w, D)
        # Add posenc (resized from base 16×16 if needed).
        posenc = _resize_posenc(self.base_posenc, h, w)       # (h, w, D)
        x = x + posenc
        # Flatten spatial and prepend CLS.
        x = x.reshape(BT, h * w, D)
        cls = self.cls_token.to(x.dtype).expand(BT, 1, D)
        return torch.cat([cls, x], dim=1)                     # (B·T, 1 + h·w, D)

    def _run_encoder(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.encoder_layers:
            tokens = blk(tokens)
        return self.encoder_final_norm(tokens)

    # -- forward ---------------------------------------------------------

    def forward(self, video: torch.Tensor,
                 state: Optional[torch.Tensor] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """video: (B, T, 3, H, W) fp32 in [0, 1].

        Returns:
            features: (B, T, D, h, w) — CLS dropped, channel-first for
                      compatibility with our downstream pipeline.
            final_state: (B, 1 + h·w, D) — carry across clip boundaries
                         at inference time.
        """
        assert video.dim() == 5, f"expected (B,T,3,H,W); got {tuple(video.shape)}"
        B, T, C, H, W = video.shape
        assert C == 3 and H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, (
            f"H, W must be multiples of {PATCH_SIZE}; got {(H, W)}")
        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        N = 1 + h * w

        # (1) Encoder — fully batched over B·T.
        frames_bt = video.reshape(B * T, C, H, W)
        tokens_bt = self._tokenize(frames_bt)                 # (B·T, N, D)
        encoded_bt = self._run_encoder(tokens_bt)             # (B·T, N, D)
        encoded = encoded_bt.view(B, T, N, self.embed_dim)

        # (2) Rnn core — sequential over T.
        if state is None:
            state = encoded.new_zeros(B, N, self.embed_dim)
        features = encoded.new_empty(B, T, N, self.embed_dim)
        for t in range(T):
            out, state = self.rnn_core(encoded[:, t], state)
            features[:, t] = out

        # (3) Reshape: drop CLS, channel-first.
        patch_feats = features[:, :, 1:, :]                   # (B, T, h·w, D)
        patch_feats = patch_feats.view(B, T, h, w, self.embed_dim)
        patch_feats = patch_feats.permute(0, 1, 4, 2, 3).contiguous()
        return patch_feats, state


# ---------------------------------------------------------------------------
# Weight conversion: NPZ (JAX tree) → PyTorch state_dict
# ---------------------------------------------------------------------------


def _recover_tree(flat: dict) -> dict:
    tree = {}
    for k, v in flat.items():
        node = tree
        parts = k.split('/')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return tree


def _jax_dense_kernel_to_torch(k: np.ndarray) -> torch.Tensor:
    """JAX Dense kernel (in_dim, out_dim) → PyTorch Linear weight (out_dim, in_dim)."""
    return torch.from_numpy(np.ascontiguousarray(k.T.copy())).float()


def _jax_qkv_kernel_to_torch(k: np.ndarray) -> torch.Tensor:
    """JAX DenseGeneral Q/K/V kernel (in_dim, num_heads, head_dim)
    → PyTorch flat Linear weight (num_heads·head_dim, in_dim)."""
    in_dim, H, D = k.shape
    flat = k.reshape(in_dim, H * D)       # (in_dim, inner)
    return torch.from_numpy(np.ascontiguousarray(flat.T.copy())).float()


def _jax_qkv_bias_to_torch(b: np.ndarray) -> torch.Tensor:
    """(num_heads, head_dim) → (num_heads·head_dim,)"""
    return torch.from_numpy(np.ascontiguousarray(b.reshape(-1).copy())).float()


def _jax_out_kernel_to_torch(k: np.ndarray) -> torch.Tensor:
    """JAX DenseGeneral output kernel (num_heads, head_dim, out_dim)
    → PyTorch Linear weight (out_dim, num_heads·head_dim)."""
    H, D, out_dim = k.shape
    flat = k.reshape(H * D, out_dim)
    return torch.from_numpy(np.ascontiguousarray(flat.T.copy())).float()


def _jax_out_bias_to_torch(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(b.copy())).float()


def _convert_mha_block(sd_out: dict, prefix: str, src: dict):
    """Fill sd_out for a `_MHA` module under `prefix.` from src subtree."""
    sd_out[prefix + 'query.weight'] = _jax_qkv_kernel_to_torch(src['query']['kernel'])
    sd_out[prefix + 'query.bias']   = _jax_qkv_bias_to_torch(src['query']['bias'])
    sd_out[prefix + 'key.weight']   = _jax_qkv_kernel_to_torch(src['key']['kernel'])
    sd_out[prefix + 'key.bias']     = _jax_qkv_bias_to_torch(src['key']['bias'])
    sd_out[prefix + 'value.weight'] = _jax_qkv_kernel_to_torch(src['value']['kernel'])
    sd_out[prefix + 'value.bias']   = _jax_qkv_bias_to_torch(src['value']['bias'])
    sd_out[prefix + 'out.weight']   = _jax_out_kernel_to_torch(src['out']['kernel'])
    sd_out[prefix + 'out.bias']     = _jax_out_bias_to_torch(src['out']['bias'])


def _convert_ln(sd_out: dict, prefix: str, src: dict):
    """JAX LayerNorm (`scale`, `bias`) → PyTorch (`weight`, `bias`)."""
    sd_out[prefix + 'weight'] = torch.from_numpy(np.ascontiguousarray(src['scale'].copy())).float()
    sd_out[prefix + 'bias']   = torch.from_numpy(np.ascontiguousarray(src['bias'].copy())).float()


def _convert_mlp(sd_out: dict, prefix: str, src: dict):
    sd_out[prefix + 'dense_in.weight']  = _jax_dense_kernel_to_torch(src['dense_in']['kernel'])
    sd_out[prefix + 'dense_in.bias']    = torch.from_numpy(np.ascontiguousarray(src['dense_in']['bias'].copy())).float()
    sd_out[prefix + 'dense_out.weight'] = _jax_dense_kernel_to_torch(src['dense_out']['kernel'])
    sd_out[prefix + 'dense_out.bias']   = torch.from_numpy(np.ascontiguousarray(src['dense_out']['bias'].copy())).float()


def _convert_gate_kernel(sd_out: dict, key: str, src_kernel: np.ndarray):
    """GRU gate Linear(no bias). JAX kernel (D, D) → Linear weight (D, D) [transposed]."""
    sd_out[key] = _jax_dense_kernel_to_torch(src_kernel)


def load_rvm_npz(model: RVMVideoSiamMAE, npz_path: str,
                  strict_load: bool = True, verbose: bool = True) -> None:
    """Load the NPZ ckpt into `model` (in-place).

    Drops the MAE pretrain decoder, mask_token, and detokenizer paths —
    only encoder + rnn_core + tokenizer + cls_token are consumed.
    """
    raw = dict(np.load(npz_path, allow_pickle=False))
    tree = _recover_tree(raw)

    sd_out = {}

    # -- Tokenizer (patch embed). ------------------------------------------
    # JAX kernel shape (1, 16, 16, 3, D). T-kernel=1, so squeeze to (16, 16, 3, D)
    # and permute to PyTorch Conv2d (D, 3, 16, 16).
    conv_k = tree['tokenizer']['patch_embedding']['Conv_0']['kernel']
    conv_k = np.squeeze(conv_k, axis=0)                # (16, 16, 3, D)
    conv_k = np.transpose(conv_k, (3, 2, 0, 1))         # (D, 3, 16, 16)
    sd_out['patch_embed.weight'] = torch.from_numpy(np.ascontiguousarray(conv_k.copy())).float()
    conv_b = tree['tokenizer']['patch_embedding']['Conv_0']['bias']
    sd_out['patch_embed.bias'] = torch.from_numpy(np.ascontiguousarray(conv_b.copy())).float()

    # -- CLS token. --------------------------------------------------------
    sd_out['cls_token'] = torch.from_numpy(np.ascontiguousarray(tree['cls_token'].copy())).float()

    # -- Encoder layers. ---------------------------------------------------
    enc = tree['encoder']
    for i in range(len(model.encoder_layers)):
        src = enc[f'layers_{i}']
        prefix = f'encoder_layers.{i}.'
        _convert_ln(sd_out, prefix + 'attention_norm.', src['attention_norm'])
        _convert_mha_block(sd_out, prefix + 'attention.', src['attention'])
        _convert_ln(sd_out, prefix + 'mlp_norm.', src['mlp_norm'])
        _convert_mlp(sd_out, prefix + 'mlp.', src['mlp'])

    # Final encoder LayerNorm. In the npz it's named "LayerNorm_0".
    _convert_ln(sd_out, 'encoder_final_norm.', enc['LayerNorm_0'])

    # -- Rnn core. ---------------------------------------------------------
    rc = tree['rnn_core']
    _convert_gate_kernel(sd_out, 'rnn_core.input_update.weight', rc['input_update']['kernel'])
    _convert_gate_kernel(sd_out, 'rnn_core.input_reset.weight',  rc['input_reset']['kernel'])
    _convert_gate_kernel(sd_out, 'rnn_core.state_update.weight', rc['state_update']['kernel'])
    _convert_gate_kernel(sd_out, 'rnn_core.state_reset.weight',  rc['state_reset']['kernel'])
    # state_layer_norm — JAX has only `scale` (no bias). PyTorch LN has bias;
    # we set weight=scale, bias=0 (and froze bias in __init__).
    sd_out['rnn_core.state_layer_norm.weight'] = torch.from_numpy(
        np.ascontiguousarray(rc['state_layer_norm']['scale'].copy())).float()
    sd_out['rnn_core.state_layer_norm.bias'] = torch.zeros(model.embed_dim, dtype=torch.float32)

    # Transformer (xa_blocks + output_norm).
    tfm = rc['transformer']
    for i in range(len(model.rnn_core.transformer.xa_blocks)):
        src = tfm[f'xa_blocks_{i}']
        p = f'rnn_core.transformer.xa_blocks.{i}.'
        _convert_ln(sd_out, p + 'ca_attention_norm.', src['ca_attention_norm'])
        _convert_mha_block(sd_out, p + 'ca_attention.', src['ca_attention'])
        _convert_ln(sd_out, p + 'mlp_norm.', src['mlp_norm'])
        _convert_mlp(sd_out, p + 'mlp.', src['mlp'])
        _convert_ln(sd_out, p + 'attention_norm.', src['attention_norm'])
        _convert_mha_block(sd_out, p + 'attention.', src['attention'])
    _convert_ln(sd_out, 'rnn_core.transformer.output_norm.', tfm['output_norm'])

    # -- Load. -------------------------------------------------------------
    missing, unexpected = model.load_state_dict(sd_out, strict=False)
    if verbose:
        print(f"[RVM] loaded {npz_path} ({len(sd_out)} tensors)")
        # base_posenc is a non-persistent buffer — expected "missing"
        missing_keys = [k for k in missing if k != 'base_posenc']
        if missing_keys:
            print(f"[RVM] unexpected MISSING keys: {missing_keys[:8]}"
                  + (f" ... (+{len(missing_keys)-8} more)" if len(missing_keys) > 8 else ""))
        if unexpected:
            print(f"[RVM] UNEXPECTED keys in state_dict: {unexpected[:8]}"
                  + (f" ... (+{len(unexpected)-8} more)" if len(unexpected) > 8 else ""))
    if strict_load and any(k != 'base_posenc' for k in missing):
        raise RuntimeError(f"RVM ckpt load missing keys: {[k for k in missing if k != 'base_posenc']}")


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_rvm(variant: str = 'small', ckpt_path: Optional[str] = None,
               device: str = 'cuda') -> RVMVideoSiamMAE:
    """Build a frozen RVM model and optionally load weights from NPZ."""
    model = RVMVideoSiamMAE(variant=variant).to(device)
    if ckpt_path is not None:
        load_rvm_npz(model, ckpt_path)
    # Freeze: RVM is a foundation model, not trained here.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
