"""CPU checks that the shims equal naive reference math. Run: python test_shims.py"""
import torch

import magi_shims as S

torch.manual_seed(0)


def naive_attn(q, k, v):
    h, hk = q.shape[2], k.shape[2]
    k = k.repeat_interleave(h // hk, dim=2)
    v = v.repeat_interleave(h // hk, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.shape[-1] ** 0.5
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)


def test_gqa():
    q, k, v = torch.randn(1, 10, 24, 16), torch.randn(1, 12, 8, 16), torch.randn(1, 12, 8, 16)
    assert torch.allclose(S.flash_attn_func(q, k, v), naive_attn(q, k, v), atol=1e-5)


def test_varlen():
    q = torch.randn(7, 4, 8)
    k = torch.randn(9, 4, 8)
    v = torch.randn(9, 4, 8)
    cq, ck = torch.tensor([0, 3, 7]), torch.tensor([0, 5, 9])
    out = S.flash_attn_varlen_func(q, k, v, cq, ck, 4, 5)
    ref = torch.cat([
        naive_attn(q[0:3][None], k[0:5][None], v[0:5][None])[0],
        naive_attn(q[3:7][None], k[5:9][None], v[5:9][None])[0],
    ])
    assert torch.allclose(out, ref, atol=1e-5)


def test_qkvpacked():
    qkv = torch.randn(2, 6, 3, 4, 8)
    ref = naive_attn(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    assert torch.allclose(S.flash_attn_qkvpacked_func(qkv), ref, atol=1e-5)


def test_rotary_matches_complex():
    b, s, h, d = 2, 5, 3, 8
    x = torch.randn(b, s, h, d)
    ang = torch.randn(s, d // 2)
    out = S.apply_rotary_emb(x, ang.cos(), ang.sin())
    z = torch.complex(x[..., : d // 2], x[..., d // 2:]) * torch.polar(torch.ones_like(ang), ang)[None, :, None]
    ref = torch.cat([z.real, z.imag], dim=-1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rotary_partial_passthrough():
    x = torch.randn(1, 4, 2, 12)
    ang = torch.randn(4, 4)  # rotates only first 8 dims
    out = S.apply_rotary_emb(x, ang.cos(), ang.sin())
    assert torch.equal(out[..., 8:], x[..., 8:]) and out.shape == x.shape


def test_range_mod():
    s, b, h, n = 6, 2, 4, 3
    x = torch.randn(s, b, h)
    gat = torch.randn(b, n, h)
    cm = torch.randint(0, n, (s, b)) + (torch.arange(b) * n)[None]  # flattened index per batch
    out = S.range_mod(x, cm, gat)
    flat = gat.flatten(0, 1)
    for si in range(s):
        for bi in range(b):
            assert torch.allclose(out[si, bi], x[si, bi] * flat[cm[si, bi]])


def test_silu_and_mul():
    x = torch.randn(3, 10)
    assert torch.allclose(S.silu_and_mul(x), torch.nn.functional.silu(x[:, :5]) * x[:, 5:])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
