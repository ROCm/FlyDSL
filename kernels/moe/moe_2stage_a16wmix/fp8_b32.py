# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""The fp8 MoE layer with 32x32-block weight scales, from bf16 tokens.

An fp8 (E4M3) checkpoint stored in 128x128 blocks whose TP shard splits a block (Qwen3.8-Flash-Next's 640-row
intermediate at TP4: 160 rows a rank) is served by vLLM's Triton ``fused_moe`` on scales refined to 32x32 blocks
(``refine_fp8_moe_block_shape``), with the weights in plain row-major ``[E, N, K]``. ``compile_moe_fp8_b32`` runs that
layer over sorted routes on the a16wmix pipeline, its weights upconverted in the kernel:

1. stage 1 (``compile_gemm1_a16w4_port(w_dtype="fp8")``): ``silu(x W1_gate^T) * (x W1_up^T)`` per route from the bf16
   tokens, each weight's fp8 code times its block's scale rounded once to bf16, fp32 accumulation, the bf16
   intermediate by sorted row (rows padded to stage 2's K tile; stage 2 masks the padding);
2. stage 2 (``compile_gemm2_a16w4_port(w_dtype="fp8")``): ``sum_route w * (h W2^T)`` with bf16 atomics into a zeroed
   output, W2's rows read to their real length.

The tilings are picked from the routing block size (``block_m``) and the layer's dims; the only token-count gate is a
16-column stage-1 tile for batches of at most four tokens at ``block_m`` 16. ``compile_moe_fp8_b32``'s docstring
states them.
"""

import functools

import torch

from kernels.common.tensor_shim import _run_compiled
from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.utils import a16wmix_resolve_arch

#: Stage 2's K tile: the intermediate's rows are padded to a multiple of it.
STAGE2_TILE_K = 128

#: Stage 1 tile at ``block_m`` 16 for a batch small enough that one CTA per (expert block, 16 gate + 16 up columns)
#: leaves the SMs underfed: 16 gate/up columns a block with four K-waves, i.e. 2x the CTAs at half the weight rows each.
SMALL_TILE_N = 16
SMALL_K_WAVE = 4


def padded_inter(inter_dim):
    """The intermediate's row length: the inter dim rounded up to stage 2's K tile."""
    return -(-int(inter_dim) // STAGE2_TILE_K) * STAGE2_TILE_K


@functools.cache
def compile_moe_fp8_b32(*, model_dim, inter_dim, experts, topk, block_m):
    """The layer's launcher for one shape class: ``launch(out, x, w1, w1_scale, w2, w2_scale, sorted_token_ids,
    sorted_weights, sorted_expert_ids, num_valid_ids, tokens, stream=None)`` overwrites ``out``.

    - ``x`` bf16 ``[tokens, model_dim]``; ``out`` bf16 ``[tokens, model_dim]`` (every row written, zero where no route
      of this rank lands);
    - ``w1`` fp8 ``[experts, 2 * inter_dim, model_dim]`` row-major (gate rows, then up rows), ``w1_scale`` f32
      ``[experts, 2 * inter_dim / 32, model_dim / 32]``; ``w2`` fp8 ``[experts, model_dim, inter_dim]``, ``w2_scale``
      f32 ``[experts, model_dim / 32, inter_dim / 32]``: value = code x its 32x32 block's scale;
    - routes sorted as aiter's ``moe_sorting`` writes them in blocks of ``block_m`` rows (``slot << 24 | token``,
      padding ``topk << 24 | tokens``), ``sorted_expert_ids`` per block, ``num_valid_ids = [rows in used blocks,
      tokens]``.

    Tile choices by regime (block size) and the layer's dims:

    - stage 1 runs one 32-column N tile a CTA at ``block_m`` 16 (``inter_dim / 32`` CTAs per routing block; the
      at-most-four-token case below uses 16 columns): the stock's whole-inter-dim tile launched one CTA per expert
      row block and left most CUs idle; 32 keeps ~5x more CTAs in flight at the same weight traffic. These tiles'
      weights go through LDS with 16-byte-per-lane coalesced copies rather than the transposed per-lane gather,
      whose every lane touches its own 32-byte sector; the LDS-B path is ``block_m`` 16 only;
    - at ``block_m`` 32/64 with ``inter_dim >= 128``, stage 1 runs a 64-column main pass plus a 32-column remainder
      when ``inter_dim % 64 == 32`` (fewer activation re-reads per weight load); ``inter_dim < 128`` uses 32-column
      tiles at every block size;
    - stage 2 contracts the real inter dim in one ``inter_dim``-wide K tile at ``block_m`` 64 and
      ``inter_dim <= 512`` instead of padding it up to ``inter_pad`` (the inter dim rounded up to 128). Otherwise it
      reads the padded row in 256-wide K tiles when ``inter_pad % 256 == 0`` and in 128-wide K tiles otherwise;
    - at ``block_m`` 16 with ``inter_pad > inter_dim`` (``inter_dim`` not a multiple of 128) stage 1 zeroes the
      intermediate's K padding and the whole output from its own grid, so the layer is two kernel launches instead of
      four; otherwise (``inter_pad == inter_dim``) there is no K padding to zero and one ``out.zero_()`` zeroes the
      output before stage 2;
    - at ``block_m`` 16, a batch of at most four tokens uses a 16-column, four-K-wave stage-1 tile;
    - the grids only cover the routes' blocks (``tokens * topk`` is an upper bound on the used blocks), not the whole
      sorted buffer.

    Value-dependent caller preconditions (not validated here: reading ``num_valid_ids`` on the host would synchronise
    and break CUDA-graph capture). ``num_valid_ids[0]`` is the number of valid sorted rows and must be a multiple of
    ``block_m`` (the kernels run whole blocks only), at most ``sorted_token_ids.numel()`` and
    ``sorted_weights.numel()`` (the per-row buffers the kernels index), and at most
    ``block_m * min(sorted_expert_ids.numel(), tokens * topk)`` (the launch grid covers that many blocks). The sorter
    writes the used blocks first, as aiter's ``moe_sorting`` does.
    """
    model_dim, inter_dim, experts, topk, block_m = (
        int(model_dim),
        int(inter_dim),
        int(experts),
        int(topk),
        int(block_m),
    )
    if model_dim % 512 or inter_dim % 32:
        raise ValueError(f"model_dim {model_dim} must be a multiple of 512 and inter_dim {inter_dim} of 32")
    if block_m not in (16, 32, 64):
        raise ValueError(f"block_m {block_m} must be 16, 32 or 64")
    if experts < 1 or topk < 1:
        raise ValueError(f"experts and topk must be >= 1, got {experts} and {topk}")
    arch = a16wmix_resolve_arch()
    if not str(arch).startswith("gfx950"):
        raise NotImplementedError(f"fp8 32x32-block MoE requires gfx950 (got {arch!r})")
    inter_pad = padded_inter(inter_dim)
    tile_n2 = 128

    # Stage 2 contracts the real inter dim in one K tile at block_m 64 / inter_dim <= 512 (no padded columns to
    # materialise or read). Otherwise it reads the padded row in 256-wide K tiles when inter_pad is a multiple of 256
    # and in 128-wide ones otherwise (inter_pad is a multiple of 128). `w_k` bounds the weight rows at the real inter
    # dim either way.
    direct_k = block_m == 64 and inter_dim <= 512
    a2_stride = inter_dim if direct_k else inter_pad
    tile_k2 = inter_dim if direct_k else (256 if inter_pad % 256 == 0 else STAGE2_TILE_K)

    # Stage 1's N tiling: one 32-column tile a CTA at block_m 16 (and at inter_dim < 128); otherwise a 64-column
    # main pass plus a 32-column remainder when inter_dim is not a multiple of 64. The parts write disjoint
    # intermediate columns.
    if block_m == 16 or inter_dim < 128:
        s1_parts = [(32, 0, inter_dim // 32)]
    else:
        _main = (inter_dim // 64) * 64
        s1_parts = [(_tn, 0, _main // _tn) for _tn in (64,)]
        if inter_dim > _main:
            s1_parts.append((32, _main, (inter_dim - _main) // 32))

    # Stage 1 zeroes the intermediate's K padding (stage 2 may read it through a fragment) and the whole output
    # (stage 2 accumulates into it with bf16 atomics) from its own grid: two launches instead of four; an extra
    # launch is ~1.8 us here, 3-10% of a 20-70 us layer. Only decode blocks: a prefill-sized output is streamed
    # better by torch's own fill than by the routing grid.
    zero_out = block_m == 16 and inter_pad > inter_dim

    def _build_gemm1(tile_n, n_offset, n_tiles, k_wave):
        return compile_gemm1_a16w4_port(
            BM=block_m,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            NE=experts,
            TOPK=topk,
            TILE_N=tile_n,
            TILE_K=128,
            act="silu",
            b_cache_mod=(1 if tile_n == 64 else 0),
            xcd_swizzle=(0 if block_m == 16 else 2),
            waves_per_eu=None,
            w_dtype="fp8",
            w_layout="standard",
            k_wave=k_wave,
            n_offset=n_offset,
            n_tiles=n_tiles,
            rocm_arch=arch,
            out_stride=a2_stride,
            use_lds_b=(block_m == 16),
            zero_out_final=zero_out,
        )

    gemm1s = [_build_gemm1(tile_n, n_offset, n_tiles, 2) for (tile_n, n_offset, n_tiles) in s1_parts]
    # At block_m 16, a 4-token batch is 40 routes over 512 experts: one 16-column tile a CTA with four K-waves (2x
    # the CTAs at half the weight rows each) keeps more CUs fed than the 32-column tile.
    gemm1_small = _build_gemm1(SMALL_TILE_N, 0, inter_dim // SMALL_TILE_N, SMALL_K_WAVE) if block_m == 16 else None

    gemm2 = compile_gemm2_a16w4_port(
        BM=block_m,
        NE=experts,
        N_OUT=model_dim,
        D_INTER=(inter_dim if direct_k else inter_pad),
        TILE_N=tile_n2,
        TILE_K=tile_k2,
        xcd_swizzle=0,
        b_cache_mod=0,
        waves_per_eu=None,
        w_dtype="fp8",
        persist=False,
        rocm_arch=arch,
        epilog="atomic",
        topk=1,
        w_k=inter_dim,
        a_stride=(a2_stride if direct_k else None),
    )

    def launch(
        out,
        x,
        w1,
        w1_scale,
        w2,
        w2_scale,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        tokens,
        stream=None,
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.dim() != 2 or x.shape[1] != model_dim:
            raise ValueError(f"x must be contiguous bf16 [{tokens}, {model_dim}]")
        if x.shape[0] != tokens or out.shape != (tokens, model_dim):
            raise ValueError(f"x/out must have {tokens} rows and {model_dim} columns")
        if out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise ValueError("out must be contiguous bf16")
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise ValueError("w1/w2 must be torch.float8_e4m3fn (gfx950 OCP e4m3)")
        if not w1.is_contiguous() or w1.shape != (experts, 2 * inter_dim, model_dim):
            raise ValueError(f"w1 must be contiguous [{experts}, {2 * inter_dim}, {model_dim}]")
        if not w2.is_contiguous() or w2.shape != (experts, model_dim, inter_dim):
            raise ValueError(f"w2 must be contiguous [{experts}, {model_dim}, {inter_dim}]")
        if w1_scale.dtype != torch.float32 or w1_scale.shape != (experts, 2 * inter_dim // 32, model_dim // 32):
            raise ValueError(f"w1_scale must be f32 [{experts}, {2 * inter_dim // 32}, {model_dim // 32}]")
        if w2_scale.dtype != torch.float32 or w2_scale.shape != (experts, model_dim // 32, inter_dim // 32):
            raise ValueError(f"w2_scale must be f32 [{experts}, {model_dim // 32}, {inter_dim // 32}]")
        for name, buf, dtype in (
            ("sorted_token_ids", sorted_token_ids, torch.int32),
            ("sorted_weights", sorted_weights, torch.float32),
            ("sorted_expert_ids", sorted_expert_ids, torch.int32),
            ("num_valid_ids", num_valid_ids, torch.int32),
        ):
            if buf.dtype != dtype or not buf.is_contiguous() or buf.dim() != 1:
                raise ValueError(f"{name} must be a contiguous 1-D {dtype} tensor")
        if num_valid_ids.numel() < 1:
            raise ValueError("num_valid_ids must hold at least the used-row count")
        if not (w1_scale.is_contiguous() and w2_scale.is_contiguous()):
            raise ValueError("weight scales must be contiguous")
        with torch.cuda.stream(stream):
            if tokens == 0:
                out.zero_()
                return out
            rows = int(sorted_token_ids.numel())
            max_m_blocks = int(sorted_expert_ids.numel())
            if rows == 0 or max_m_blocks == 0:
                raise ValueError("sorted_token_ids/sorted_expert_ids must be non-empty for tokens > 0")
            inter = torch.empty(rows, a2_stride, dtype=torch.bfloat16, device=x.device)
            if not zero_out:
                out.zero_()
        # No more m-blocks than routes: every used expert owns at most one block per `block_m` routes, and the sorter
        # puts the used blocks first. The buffers are sized for the whole model's expert count.
        blocks = min(max_m_blocks, int(tokens) * int(topk))
        small = gemm1_small is not None and int(tokens) <= 4
        for (tile_n, _n_offset, n_tiles), g1 in zip(s1_parts, gemm1s):
            g, tn, nt = (gemm1_small, SMALL_TILE_N, inter_dim // SMALL_TILE_N) if small else (g1, tile_n, n_tiles)
            grid1 = gemm1_a16w4_grid(block_m, INTER=inter_dim, TILE_N=tn, max_m_blocks=blocks, n_tiles=nt)
            args = (
                x.data_ptr(),
                w1.data_ptr(),
                w1_scale.data_ptr(),
                sorted_expert_ids.data_ptr(),
                num_valid_ids.data_ptr(),
                sorted_token_ids.data_ptr(),
                int(tokens),
                int(grid1),
                1.0,
                1.0,
                1.0,
                1.0,
                float("inf"),
                inter.data_ptr(),
                stream,
            )
            if zero_out:
                args = args + (int(out.data_ptr()),)
            _run_compiled(g, *args)
        grid2 = gemm2_a16w4_grid(block_m, N_OUT=model_dim, TILE_N=tile_n2, max_m_blocks=blocks, persist=False)
        _run_compiled(
            gemm2,
            inter.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            int(tokens),
            int(max_m_blocks),
            int(grid2),
            out.data_ptr(),
            stream,
        )
        return out

    return launch
