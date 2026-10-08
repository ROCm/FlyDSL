# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared helpers of the attention tests: the fp64 reference and its rounding floor, NaN-poisoned allocation
with the 8xD slack, layout round-robin, varlen builders, and the launch wrappers.

The reference and the gate follow AOTriton's `test_common_mistakes`: an fp64 attention that rounds to the input dtype
**only where a correct flash kernel has to** (P before P@V, dS before dS@K and dS^T@Q, each stored output). Its
distance from the exact answer is the error floor: what a kernel that keeps everything else in fp32 still gets. Gates
are against that floor (G-floor), not against torch's low-precision SDPA with fudge factors, which cannot see a
precision mistake.
"""

import itertools
import math

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
DTYPE_STRS = {torch.bfloat16: "bf16", torch.float16: "f16"}

# A correct kernel lands at or below 1.0x the floor; the mistakes this suite exists for measured 2.6x-17x.
FLOOR_MULT = 2.0
LSE_REL_BOUND = 2.0**-16

WINDOW_TOPLEFT = -2147483647
WINDOW_BOTRIGHT = -2147483646

PERMS = list(itertools.permutations(range(3)))  # physical order of the (B, H, S) axes


def ceil8(n):
    return (n + 7) // 8 * 8


def seeded(seed=0):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


# ---------------------------------------------------------------------------
# Allocation: NaN slack, layout round-robin
# ---------------------------------------------------------------------------


def alloc(b, h, s, d, dtype, perm=(0, 1, 2), fill=None, slack=True, poison=float("nan")):
    """A BHSD-shaped `[..., :d]` view of an allocation whose D axis is padded to `ceil8(d)` and whose outer axes are
    laid out in `perm` order (physical). The slack (and, with `fill=None`, everything) is **NaN**, never a finite
    number: `torch.empty_like` on a narrowed view flattens the layout and drops the slack, and a zeroed slack would
    hide a kernel that reads or writes it. `poison` is what the slack holds (default NaN)."""
    dims = (b, h, s)
    dp = ceil8(d) if slack else d
    phys = [dims[i] for i in perm] + [dp]
    buf = torch.full(phys, poison, device="cuda", dtype=dtype)
    inv = [0] * 3
    for pos, axis in enumerate(perm):
        inv[axis] = pos
    view = buf.permute(*inv, 3)[..., :d]
    if fill is not None:
        view.copy_(fill)
    return view


def randn(b, h, s, d, dtype, perm=(0, 1, 2), scale=1.0, gen=None, poison=float("nan")):
    data = (torch.randn(b, h, s, d, device="cuda", generator=gen, dtype=torch.float32) * scale).to(dtype)
    return alloc(b, h, s, d, dtype, perm, fill=data, poison=poison)


def layouts(n, offset=0):
    """`n` distinct outer-axis permutations, round-robin."""
    return [PERMS[(offset + i) % len(PERMS)] for i in range(n)]


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def window_bounds(left, right, sq, sk):
    """`(left, right)` with the causal sentinels resolved against one sequence's lengths (python ints), as
    `common.resolve_window` does on the device."""
    if left in (WINDOW_TOPLEFT, WINDOW_BOTRIGHT):
        left_r = sq
    else:
        left_r = left
    if right == WINDOW_TOPLEFT:
        right_r = 0
    elif right == WINDOW_BOTRIGHT:
        right_r = sk - sq
    else:
        right_r = right
    return left_r, right_r


def window_mask(sq, sk, left, right, device="cuda"):
    """`keep[i, j]`: `i - left <= j <= i + right`, sentinels resolved."""
    lo, hi = window_bounds(left, right, sq, sk)
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return (j <= i + hi) & (j >= i - lo)


def causal_mask(sq, sk, bottom_right=True, device="cuda"):
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + ((sk - sq) if bottom_right else 0)


def alibi_bias(slopes, b, h, sq, sk, device="cuda"):
    """The ALiBi term as an additive `(B, H, Sq, Sk)` fp64 bias, in natural units: `-slope * |i + (sk - sq) - j|`,
    bottom-right aligned. `slopes` is `(H,)` (shared) or `(B, H)`."""
    s = slopes.detach().to(torch.float64)
    s = s.expand(b, h) if s.dim() == 1 else s
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    rel = (i + (sk - sq) - j).abs().to(torch.float64)
    return -s[:, :, None, None] * rel[None, None]


def reference(q, k, v, sm_scale, *, mask=None, bias=None, keep=None, p_drop=0.0, round_to=None, ftz=False, sink=None):
    """fp64 attention forward from the same low-precision inputs: `(o, lse)`.

    `lse` is natural-base and `-inf` for a fully masked row (the kernel stores `+inf`). `mask` is boolean,
    broadcastable to `(B, H, Sq, Sk)`. With `round_to=None` this is the exact answer; with a dtype it rounds where a
    correct kernel must, so its distance from the exact one is the floor. `keep` (same shape) is a dropout keep mask:
    the softmax denominator is the *undropped* sum, and survivors are scaled `1/(1-p)` once.
    """
    q, k, v = (t.detach().to(torch.float64) for t in (q, k, v))
    hq, hk = q.shape[1], k.shape[1]
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)

    def rnd(x):
        if round_to is None:
            return x
        x = x.to(round_to).to(torch.float64)
        if ftz:
            x = torch.where(x.abs() < torch.finfo(round_to).tiny, 0.0, x)
        return x

    s = (q @ k.transpose(-1, -2)) * sm_scale
    if bias is not None:
        s = s + bias.detach().to(torch.float64)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    if sink is not None:
        # An attention sink is one extra logit per head that joins the softmax denominator and has no value: it takes
        # weight from the keys, and it makes every row live (a fully masked row has LSE == sink and O == 0).
        extra = sink.detach().to(torch.float64).view(1, -1, 1, 1).expand(s.shape[0], s.shape[1], s.shape[2], 1)
        lse = torch.logsumexp(torch.cat([s, extra], dim=-1), dim=-1, keepdim=True)
    else:
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
    live = torch.isfinite(lse)
    p = torch.where(live, torch.exp(s - torch.where(live, lse, torch.zeros_like(lse))), torch.zeros_like(s))
    if keep is not None:
        p = p * keep.to(torch.float64)
    o = rnd(rnd(p) @ v)
    if keep is not None:
        o = rnd(o / (1.0 - p_drop)) if round_to is not None else o / (1.0 - p_drop)
    return o, lse.squeeze(-1)


def floor_rel(ref_fn, exact, round_to):
    """The rounding floor of the output: the larger of the FTZ and non-FTZ models (the kernel's default DAZ flushes
    subnormal inputs; a conforming kernel can land on either side)."""
    return max(relrms(ref_fn(round_to=round_to, ftz=ftz)[0], exact) for ftz in (False, True))


def relrms(x, ref):
    x, ref = x.double(), ref.double()
    n = ref.norm()
    return ((x - ref).norm() / n).item() if n > 0 else (x - ref).norm().item()


def check_floor(name, got, exact, floor, ctx="", mult=FLOOR_MULT):
    """G-floor: relative RMS error at most `mult` x the floor; no NaN anywhere."""
    assert not torch.isnan(got).any(), f"{ctx}: {name} has NaN"
    err = relrms(got, exact)
    # A floor of exactly zero (tiny outputs, exact reference) must not make the bound unreachable.
    bound = mult * max(floor, 1e-7)
    assert err <= bound, f"{ctx}: {name} error {err:.3e} is {err / max(floor, 1e-30):.1f}x the floor {floor:.3e}"


def check_lse(got, ref_lse, live_rows, ctx=""):
    """LSE to fp32 accuracy on live rows: `2**-16 * max(1, max|S|)`; fully masked rows are **exactly +inf**."""
    got = got.double()
    if live_rows.any():
        ref_live = ref_lse[live_rows]
        s_max = max(1.0, ref_live.abs().max().item())
        err = (got[live_rows] - ref_live).abs().max().item()
        assert err <= LSE_REL_BOUND * s_max, f"{ctx}: LSE off by {err:.3e}, bound {LSE_REL_BOUND * s_max:.3e}"
    dead = ~live_rows
    if dead.any():
        bad = got[dead]
        n_nan, n_neg = int(torch.isnan(bad).sum()), int((bad == float("-inf")).sum())
        n_notinf = int((bad != float("inf")).sum())
        assert (
            n_notinf == 0
        ), f"{ctx}: masked LSE rows must be exactly +inf ({n_nan} NaN, {n_neg} -inf, {n_notinf} not +inf)"


# ---------------------------------------------------------------------------
# Launch wrappers
# ---------------------------------------------------------------------------


def run_fwd(fn, q, k, v, o, *, lse=None, scale=None, window=None, bias=None, p_drop=None, seed=0, offset=0, **kw):
    """Launch a built forward. Shapes and strides are read off the tensors; `seqlen_q/k` default to their extents."""
    b, _, sq, _ = q.shape
    extra = {}
    if p_drop is not None:
        extra.update(dropout_p=p_drop, philox_seed=seed, philox_offset2=offset)
    fn(
        q,
        k,
        v,
        o,
        b,
        sq,
        seqlen_k=k.shape[2],
        scale=scale,
        lse=lse,
        window=window,
        bias=bias,
        **extra,
        **kw,
    )
    torch.cuda.synchronize()
    return o


def lse_alloc(b, h, s):
    return torch.full((b, h, s), float("nan"), device="cuda", dtype=torch.float32)


def sdpa_scale(d):
    return 1.0 / math.sqrt(d)


def meta_of(**kw):
    """`FmhaInputMetadata` for the given inputs (the metadata class is the arch-neutral description)."""
    from kernels.attention.flash_attn_gfx950_config import FmhaInputMetadata

    kw.setdefault("dtype_str", "bf16")
    return FmhaInputMetadata(**kw)


def fwd_check(
    fn,
    *,
    b=2,
    hq=4,
    hk=None,
    sq=129,
    sk=None,
    d=64,
    dv=None,
    dtype,
    window=None,
    bias=None,
    scale=None,
    input_scale=1.0,
    seed=0,
    perms=None,
    ctx="",
    want_lse=True,
    mult=FLOOR_MULT,
    qkv=None,
    alibi_slopes=None,
    sink=None,
    **launch_kw,
):
    """Run one forward and gate it against the fp64 floor (G-floor); returns the pieces for further checks.

    Allocates with NaN slack and NaN-prefilled outputs, launches through the built `fn`, and checks O (relative RMS at
    most `mult` x the rounding floor, no NaN) and LSE (fp32 accuracy on live rows, exactly `+inf` on masked ones).
    `window=(left, right)` (sentinels allowed) selects the window mask; `bias` is a `(B, Hq, Sq, Sk)` tensor.
    """
    hk = hq if hk is None else hk
    sk = sq if sk is None else sk
    dv = d if dv is None else dv
    gen = seeded(seed)
    pq, pk, pv, po = perms or (PERMS[0],) * 4
    if qkv is None:
        q = randn(b, hq, sq, d, dtype, pq, scale=input_scale, gen=gen)
        k = randn(b, hk, sk, d, dtype, pk, scale=input_scale, gen=gen)
        v = randn(b, hk, sk, dv, dtype, pv, gen=gen)
    else:
        q, k, v = qkv
    o = alloc(b, hq, sq, dv, dtype, po)
    lse = lse_alloc(b, hq, sq) if want_lse else None
    sm_scale = sdpa_scale(d) if scale is None else scale
    if alibi_slopes is not None:
        launch_kw["alibi_slopes"] = alibi_slopes
    if sink is not None:
        launch_kw["sink"] = sink
    run_fwd(fn, q, k, v, o, lse=lse, scale=scale, window=window, bias=bias, **launch_kw)

    mask = window_mask(sq, sk, *window) if window is not None else None
    ref_bias = bias
    if alibi_slopes is not None:
        alibi = alibi_bias(alibi_slopes, b, hq, sq, sk)
        ref_bias = alibi if bias is None else bias.to(torch.float64) + alibi

    def ref(**kw):
        return reference(q, k, v, sm_scale, mask=mask, bias=ref_bias, sink=sink, **kw)

    exact_o, exact_lse = ref()
    floor = floor_rel(ref, exact_o, dtype)
    check_floor("O", o, exact_o, floor, ctx, mult)
    live = torch.isfinite(exact_lse)
    if lse is not None:
        check_lse(lse, exact_lse, live, ctx)
    dead = ~live
    if dead.any():
        n = int((o.permute(0, 1, 2, 3)[dead] != 0).sum())
        assert n == 0, f"{ctx}: {n} nonzero elements in fully masked O rows"
    return dict(q=q, k=k, v=v, o=o, lse=lse, exact_o=exact_o, exact_lse=exact_lse, floor=floor, ref=ref)


# ---------------------------------------------------------------------------
# Varlen
# ---------------------------------------------------------------------------

VARLEN_MODES = ("0x0B0B", "0x0202", "0x1313", "0x150B", "0x040B")
_STACKED_Q = ("0x0B0B", "0x1313", "0x150B", "0x040B")
_STACKED_K = ("0x0B0B", "0x1313", "0x150B")


class VarlenCase:
    """One varlen problem in one of the five `VarlenBits` modes, with per-sequence access to every tensor.

    `q/k/v/o` are laid out as the mode dictates (packed `(1, H, T, D)` for stacked sides, batched `(N, H, S, D)`
    otherwise); `seq(z)` gives the sequence-`z` rows of each tensor so the reference can run per sequence. NaN slack and
    NaN-filled outputs as everywhere.
    """

    def __init__(
        self, mode, lens_q, lens_k, hq, hk, d, dtype, *, dv=None, lse_layout="HT", gap=3, seed=0, input_scale=1.0
    ):
        from kernels.attention import abi

        self.mode, self.lens_q, self.lens_k = mode, list(lens_q), list(lens_k)
        self.hq, self.hk, self.d, self.dv, self.dtype = hq, hk, d, d if dv is None else dv, dtype
        n = len(lens_q)
        self.n = n
        gen = seeded(seed)
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")  # noqa: E731
        cum = lambda xs: [0] + list(itertools.accumulate(xs))  # noqa: E731
        maxq, maxk = max(lens_q), max(lens_k)
        self.maxq, self.maxk = maxq, maxk
        lse_bits = abi.VARLEN_LSE_LAYOUT_TH if lse_layout == "TH" else abi.VARLEN_LSE_LAYOUT_HT
        self.lse_layout = lse_layout
        stacked_q, stacked_k = mode in _STACKED_Q, mode in _STACKED_K
        if mode == "0x1313":
            # padding *between* sequences: positions come from their own array
            self.q_start, pos = [], 0
            for ln in lens_q:
                self.q_start.append(pos)
                pos += ln + gap
            self.tq = pos
            self.k_start, pos = [], 0
            for ln in lens_k:
                self.k_start.append(pos)
                pos += ln + gap
            self.tk = pos
        else:
            self.q_start = cum(lens_q)[:-1] if stacked_q else [0] * n
            self.k_start = cum(lens_k)[:-1] if stacked_k else [0] * n
            self.tq = sum(lens_q) if stacked_q else maxq
            self.tk = sum(lens_k) if stacked_k else maxk
        bq = 1 if stacked_q else n
        bk = 1 if stacked_k else n
        self.batch = bq
        self.num_seqlens = n if stacked_q else 0
        self.q = alloc(bq, hq, self.tq, d, dtype, fill=torch.zeros(bq, hq, self.tq, d))
        self.k = alloc(bk, hk, self.tk, d, dtype, fill=torch.zeros(bk, hk, self.tk, d))
        self.v = alloc(bk, hk, self.tk, self.dv, dtype, fill=torch.zeros(bk, hk, self.tk, self.dv))
        for z in range(n):
            qb, qr = self._where_q(z)
            kb, kr = self._where_k(z)
            self.q[qb, :, qr] = (torch.randn(hq, lens_q[z], d, device="cuda", generator=gen) * input_scale).to(dtype)
            self.k[kb, :, kr] = (torch.randn(hk, lens_k[z], d, device="cuda", generator=gen) * input_scale).to(dtype)
            self.v[kb, :, kr] = torch.randn(hk, lens_k[z], self.dv, device="cuda", generator=gen).to(dtype)
        self.o = alloc(bq, hq, self.tq, self.dv, dtype)
        # LSE is always compact: (H, T) per stacked batch (HT) / (T, H) (TH); padded sides pad every row group.
        tokens = self.tq if stacked_q else maxq
        shape = (bq * hq, tokens) if lse_layout == "HT" else (bq * tokens, hq)
        self.lse = torch.full(shape, float("nan"), device="cuda", dtype=torch.float32)
        args = dict(lse_tokens=tokens, lse_layout=lse_bits)
        if mode == "0x0B0B":
            self.varlen = abi.varlen_compact(i32(cum(lens_q)), i32(cum(lens_k)), maxq, maxk, **args)
        elif mode == "0x0202":
            self.varlen = abi.varlen_padded(i32(cum(lens_q)), i32(cum(lens_k)), maxq, maxk, **args)
        elif mode == "0x1313":
            self.varlen = abi.varlen_strided(
                i32(cum(lens_q)),
                i32(cum(lens_k)),
                i32(self.q_start + [self.tq]),
                i32(self.k_start + [self.tk]),
                maxq,
                maxk,
                **args,
            )
        elif mode == "0x150B":
            self.varlen = abi.varlen_seqused_k(i32(cum(lens_q)), i32(cum(lens_k)), i32(lens_k), maxq, maxk, **args)
        else:
            self.varlen = abi.varlen_seqused_k(i32(cum(lens_q)), None, i32(lens_k), maxq, maxk, k_is_cache=True, **args)

    def _where_q(self, z):
        return (0 if self.mode in _STACKED_Q else z), slice(self.q_start[z], self.q_start[z] + self.lens_q[z])

    def _where_k(self, z):
        return (0 if self.mode in _STACKED_K else z), slice(self.k_start[z], self.k_start[z] + self.lens_k[z])

    def seq(self, z):
        """`(q, k, v)` of sequence `z` as `(1, H, S, D)` tensors."""
        qb, qr = self._where_q(z)
        kb, kr = self._where_k(z)
        return self.q[qb : qb + 1, :, qr], self.k[kb : kb + 1, :, kr], self.v[kb : kb + 1, :, kr]

    def o_of(self, z):
        qb, qr = self._where_q(z)
        return self.o[qb : qb + 1, :, qr]

    def lse_of(self, z):
        """LSE of sequence `z` as `(H, S)`."""
        qb, qr = self._where_q(z)
        if self.lse_layout == "HT":
            return self.lse.view(self.batch, self.hq, -1)[qb][:, qr]
        return self.lse.view(self.batch, -1, self.hq)[qb][qr].transpose(0, 1)

    def launch(self, fn, **kw):
        fn(
            self.q,
            self.k,
            self.v,
            self.o,
            self.batch,
            self.maxq,
            seqlen_k=self.maxk,
            lse=self.lse,
            varlen=self.varlen,
            num_seqlens=self.num_seqlens,
            **kw,
        )
        torch.cuda.synchronize()

    def check(self, fn, *, window=None, scale=None, ctx="", mult=FLOOR_MULT, alibi_slopes=None, sink=None, bias=None, **kw):
        """Launch and gate every sequence against its own fp64 reference (O to the floor, LSE to fp32). `bias` is
        `(batch, H, total_q or max_q, cols >= max_k)` following Q's layout: a sequence's rows are its Q rows and its live
        columns are the first `seqlen_k` of them."""
        if alibi_slopes is not None:
            kw["alibi_slopes"] = alibi_slopes
        if sink is not None:
            kw["sink"] = sink
        if bias is not None:
            kw["bias"] = bias
        self.launch(fn, scale=scale, window=window, **kw)
        sm = sdpa_scale(self.d) if scale is None else scale
        for z in range(self.n):
            q, k, v = self.seq(z)
            sq, sk = q.shape[2], k.shape[2]
            mask = window_mask(sq, sk, *window) if window is not None else None

            alibi = None
            if alibi_slopes is not None:
                row = alibi_slopes if alibi_slopes.dim() == 1 else alibi_slopes[z]
                alibi = alibi_bias(row, 1, self.hq, sq, sk)
            if bias is not None:
                qb_, qr_ = self._where_q(z)
                bz = bias[qb_ : qb_ + 1, :, qr_, :sk].to(torch.float64)
                alibi = bz if alibi is None else alibi + bz

            def ref(q=q, k=k, v=v, mask=mask, alibi=alibi, **rk):
                return reference(q, k, v, sm, mask=mask, bias=alibi, sink=sink, **rk)

            ex_o, ex_lse = ref()
            o = self.o_of(z)
            assert not torch.isnan(o).any(), f"{ctx}: sequence {z} O has NaN"
            err, floor = relrms(o, ex_o), floor_rel(ref, ex_o, self.dtype)
            assert err <= mult * max(floor, 1e-7), f"{ctx}: seq {z} O error {err:.3e} vs floor {floor:.3e}"
            live = torch.isfinite(ex_lse[0])
            check_lse(self.lse_of(z), ex_lse[0], live, f"{ctx} seq {z}")
            dead = ~live
            if dead.any():
                assert int((o[0][dead] != 0).sum()) == 0, f"{ctx}: seq {z} masked O rows are not exactly 0"


def compile_inputs(meta, d=None, dtype=None):
    """`(args, kwargs)` for `launcher.compile(*args, **kwargs)`: small real tensors of the right shape and types for the
    build `meta` describes (a compile needs argument *types*, not meaningful data)."""
    d = meta.head_dim if d is None else d
    dtype = DTYPES[meta.dtype_str] if dtype is None else dtype
    q = randn(1, 2, 64, d, dtype)
    k = randn(1, 2, 64, d, dtype)
    v = randn(1, 2, 64, meta.head_dim_v_real, dtype)
    o = alloc(1, 2, 64, meta.head_dim_v_real, dtype)
    kw = {}
    if meta.window:
        kw["window"] = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
    if meta.bias:
        kw["bias"] = torch.zeros(1, 2, 64, 64, device="cuda", dtype=dtype)
    if meta.dropout:
        kw.update(dropout_p=0.5, philox_seed=1)
    if meta.alibi:
        kw["alibi_slopes"] = torch.full((2,), 0.25, device="cuda", dtype=torch.float32)
    if meta.sink:
        kw["sink"] = torch.zeros(2, device="cuda", dtype=torch.float32)
    return (q, k, v, o, 1, 64), kw
