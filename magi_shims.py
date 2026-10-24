"""Pure-PyTorch stand-ins for the compiled kernels the official MAGI-1 inference code imports.

MAGI-1 hard-imports flash_attn (pinned 2.4.2), flashinfer and triton at module level, none of
which can be built or pinned on Graydient's ephemeral runners. Every use is on the non-fp8 path
and has a plain equivalent, so we register small modules in sys.modules *before* the `inference`
package is imported. Only bf16 weights are supported (fp8 needs flashinfer.bmm_fp8).

Layout conventions follow flash_attn: q is [batch, seq, heads, dim]; k/v may have fewer heads
(GQA); varlen takes packed [total, heads, dim] plus cu_seqlens.
"""
import sys
import types

import torch
import torch.nn.functional as F


def _sdpa(q, k, v, softmax_scale=None, causal=False):
    """q [b, sq, h, d], k/v [b, sk, hk, d] -> [b, sq, h, d]."""
    h, hk = q.shape[2], k.shape[2]
    if hk != h:
        rep = h // hk
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=softmax_scale,
    )
    return out.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    q, k, v = qkv.unbind(dim=2)
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """Packed q [total_q, h, d], k/v [total_k, hk, d]; one SDPA call per sequence."""
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].unsqueeze(0)
        ki = k[ck[i]:ck[i + 1]].unsqueeze(0)
        vi = v[ck[i]:ck[i + 1]].unsqueeze(0)
        outs.append(_sdpa(qi, ki, vi, softmax_scale, causal).squeeze(0))
    return torch.cat(outs, dim=0)


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, **_):
    """flash_attn.layers.rotary.apply_rotary_emb: x [b, s, h, d], cos/sin [s, rot/2].
    Rotates the first 2*cos.shape[-1] dims; the rest pass through."""
    ro = cos.shape[-1] * 2
    cos = cos.unsqueeze(1).to(x.dtype)  # [s, 1, rot/2], broadcasts over [b, s, h, rot/2]
    sin = sin.unsqueeze(1).to(x.dtype)
    xr = x[..., :ro]
    if interleaved:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    else:
        x1, x2 = xr[..., : ro // 2], xr[..., ro // 2:]
        rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return torch.cat((rot, x[..., ro:]), dim=-1) if ro < x.shape[-1] else rot


def silu_and_mul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def bmm_fp8(*_, **__):
    raise NotImplementedError("fp8 GEMM is not available: use the bf16 MAGI-1 weights (fp8_quant=false)")


def range_mod(x, c_mapping, gatings):
    """Torch version of dit_module.range_mod_triton.
    x [s, b, h], c_mapping [s, b] (index into the flattened gatings), gatings [b, n, h]."""
    s, b, h = x.shape
    xf = x.transpose(0, 1).flatten(0, 1)
    idx = c_mapping.transpose(0, 1).flatten(0, 1).long()
    y = xf * gatings.flatten(0, 1)[idx]
    return y.reshape(b, s, h).transpose(0, 1)


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _install_fake_triton():
    """Only when triton is missing (e.g. Windows). The kernels are replaced by range_mod anyway;
    the module just has to import and let the @triton.jit definitions evaluate."""
    def jit(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

    tl = _module("triton.language", constexpr=int, program_id=None, arange=None, load=None, store=None)
    _module("triton", jit=jit, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(), language=tl)


def install():
    """Register the fake flash_attn / flashinfer (and triton if absent). Idempotent."""
    if getattr(sys.modules.get("flash_attn"), "_magi_shim", False):
        return
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        _install_fake_triton()

    fa = _module("flash_attn", _magi_shim=True, flash_attn_func=flash_attn_func,
                 flash_attn_varlen_func=flash_attn_varlen_func,
                 flash_attn_qkvpacked_func=flash_attn_qkvpacked_func)
    fa.flash_attn_interface = _module("flash_attn.flash_attn_interface", flash_attn_func=flash_attn_func)
    layers = _module("flash_attn.layers")
    layers.rotary = _module("flash_attn.layers.rotary", apply_rotary_emb=apply_rotary_emb)
    fa.layers = layers

    fi = _module("flashinfer", _magi_shim=True)
    fi.gemm = _module("flashinfer.gemm", bmm_fp8=bmm_fp8)
    fi.activation = _module("flashinfer.activation", silu_and_mul=silu_and_mul)


def patch_dit_module(dit_module):
    """Swap the triton-launching function for the torch one (call after importing dit_module)."""
    dit_module.range_mod_triton = range_mod
