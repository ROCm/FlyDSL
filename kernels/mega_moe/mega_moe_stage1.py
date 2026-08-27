# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Protocol router for MegaMoE Stage1 kernels."""

from flydsl.runtime.device import get_rocm_arch  # re-exported for test overrides

from .mega_moe_config import A8W4_DECODE_MTPRS, A8W4SMOOTH_DECODE_MTPRS
__all__ = [
    "A8W4_ENTRY_COUNT_SHARDS",
    "ENTRY_COUNT_SHARDS",
    "ENTRY_EPOCH_SLOT_COUNT",
    "compile_mega_moe_stage1",
    "get_rocm_arch",
    "run_mega_moe_stage1",
]

_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
_GRID_EPOCH_DISPATCH_STRIDE = 256
ENTRY_COUNT_SHARDS = 16
A8W4_ENTRY_COUNT_SHARDS = 256
ENTRY_EPOCH_SLOT_COUNT = len(_GRID_MULT_VALUES) * (
    1 + _GRID_EPOCH_DISPATCH_STRIDE
)


def _entry_epoch_slot(
    grid_mult: int, dispatch_blocks: int, retire_control_roles: bool
) -> int:
    """Key persistent launch state by the CTA population that advances it."""
    grid_mult_index = _GRID_MULT_VALUES.index(int(grid_mult))
    if not retire_control_roles:
        return grid_mult_index
    if not 0 < int(dispatch_blocks) < _GRID_EPOCH_DISPATCH_STRIDE:
        raise ValueError(f"dispatch_blocks out of range: {dispatch_blocks}")
    return (
        len(_GRID_MULT_VALUES)
        + grid_mult_index * _GRID_EPOCH_DISPATCH_STRIDE
        + int(dispatch_blocks)
    )


def _stage1_quant_traits(quant_mode: str, inter_dim: int, tile_n: int):
    """Return compile-time Stage1 geometry for a quant mode."""
    if quant_mode not in ("a8w4", "a8w4smooth", "w8a8smooth"):
        raise ValueError(f"unsupported Stage1 quant_mode={quant_mode!r}")
    is_int8 = quant_mode in ("a8w4smooth", "w8a8smooth")
    gemm_n = int(inter_dim) if is_int8 else 2 * int(inter_dim)
    if tile_n <= 0 or gemm_n % int(tile_n):
        raise ValueError(
            f"Stage1 N={gemm_n} must tile evenly by tile_n={tile_n}"
        )
    return {
        "is_int8": is_int8,
        "packed_int4": quant_mode == "a8w4smooth",
        "gemm_n": gemm_n,
        "n_tiles": gemm_n // int(tile_n),
        "uses_mx_scale_lds": not is_int8,
    }


def _validate_a8w4_decode_contract(
    *, model_dim, inter_dim, experts_per_rank, fuse_npes, fuse_topk,
    fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, external_grouping,
    external_counting, payload_chunk_rows, payload_tile_ready, work_shards,
):
    shape = (model_dim, inter_dim, experts_per_rank, fuse_npes, fuse_topk)
    if shape != (3584, 1280, 48, 8, 8) or fuse_scale_dim != 112:
        raise ValueError(
            "A8W4 MX-scale Stage1 is decode-only for M13 "
            "(D=3584, I=1280, EPR=48, EP=8, topk=8, scale_dim=112)"
        )
    if fuse_mtpr not in A8W4_DECODE_MTPRS:
        raise ValueError(
            f"A8W4 MX-scale decode requires MTPR in {A8W4_DECODE_MTPRS}, got {fuse_mtpr}"
        )
    expected_fixed = fuse_mtpr <= 128
    if bool(fixed_slot_dispatch) != expected_fixed:
        path = "fixed-slot" if expected_fixed else "bounded compact"
        raise ValueError(f"A8W4 MX-scale MTPR={fuse_mtpr} requires {path} dispatch")
    advanced = (
        bool(external_grouping), bool(external_counting),
        int(payload_chunk_rows), bool(payload_tile_ready),
    )
    if any(advanced):
        raise ValueError(
            "A8W4 MX-scale decode does not support external grouping/counting, "
            "or chunked/tile-ready payloads"
        )
    if work_shards is not None:
        raise ValueError("A8W4 MX-scale decode has no dynamic work-shard queue")


def _validate_a8w4smooth_decode(common, *, mxfp4_transport, smoothquant_mode):
    """Pin the only supported A8W4 Smooth contract before JIT specialization."""
    if mxfp4_transport:
        raise ValueError("a8w4smooth decode does not support MXFP4 transport")
    if smoothquant_mode != "none":
        raise ValueError("a8w4smooth decode requires standalone front SmoothQuant")
    if (
        common["model_dim"],
        common["inter_dim"],
        common["experts_per_rank"],
        common["fuse_npes"],
        common["fuse_topk"],
        common["fuse_scale_dim"],
    ) != (3584, 1280, 48, 8, 8, 4):
        raise ValueError(
            "a8w4smooth decode is specialized for M13 EP8 with one f32 scale per route"
        )
    if not common["fixed_slot_dispatch"]:
        raise ValueError("a8w4smooth decode requires fixed-slot dispatch")
    if common["fuse_mtpr"] not in A8W4SMOOTH_DECODE_MTPRS:
        raise ValueError(
            "a8w4smooth decode requires MTPR in "
            "{1,4,8,16,32,64,128,256,512}"
        )
    if any(
        (
            common["work_shards"] is not None,
            common["external_grouping"],
            common["external_counting"],
            common["payload_chunk_rows"],
            common["payload_tile_ready"],
        )
    ):
        raise ValueError(
            "a8w4smooth decode dynamic work queues and advanced dispatch flags "
            "must be disabled"
        )


# fmt: off
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    work_shards: int | None = None, external_grouping: bool | None = None,
    external_counting: bool | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    swiglu_limit: float = 0.0,
    quant_mode: str = "a8w4", mxfp4_transport: bool = False,
    smoothquant_mode: str = "none",
):
    common = dict(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank,
        experts_per_rank=experts_per_rank, fuse_npes=fuse_npes,
        fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim,
        fixed_slot_dispatch=fixed_slot_dispatch, sort_block_m=sort_block_m,
        tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights,
        mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu,
        num_dispatch_cu=num_dispatch_cu, b_nt=b_nt,
        swiglu_limit=swiglu_limit,
    )
    if quant_mode == "a8w4":
        if mxfp4_transport:
            raise ValueError("MXFP4 transport requires a SmoothQuant compute mode")
        if smoothquant_mode != "none":
            raise ValueError("fused SmoothQuant prepare is not an A8W4 MX-scale mode")
        _validate_a8w4_decode_contract(
            model_dim=model_dim, inter_dim=inter_dim,
            experts_per_rank=experts_per_rank, fuse_npes=fuse_npes,
            fuse_topk=fuse_topk, fuse_mtpr=fuse_mtpr,
            fuse_scale_dim=fuse_scale_dim,
            fixed_slot_dispatch=fixed_slot_dispatch,
            external_grouping=external_grouping,
            external_counting=external_counting,
            payload_chunk_rows=payload_chunk_rows,
            payload_tile_ready=payload_tile_ready,
            work_shards=work_shards,
        )
        from .mega_moe_stage1_main_a8w4 import compile_mega_moe_stage1 as compile_a8w4

        return compile_a8w4(**common)
    if quant_mode not in ("a8w4smooth", "w8a8smooth"):
        raise ValueError(f"unsupported Stage1 quant_mode={quant_mode!r}")
    from .mega_moe_stage1_smooth import compile_mega_moe_stage1 as compile_smooth

    smooth_common = dict(
        common,
        work_shards=work_shards,
        external_grouping=external_grouping,
        external_counting=external_counting,
        payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
    )
    if quant_mode == "a8w4smooth":
        _validate_a8w4smooth_decode(
            smooth_common,
            mxfp4_transport=mxfp4_transport,
            smoothquant_mode=smoothquant_mode,
        )
    else:
        if smooth_common["fixed_slot_dispatch"]:
            raise ValueError("w8a8smooth prefill requires compact dispatch")

    return compile_smooth(
        **smooth_common,
        quant_mode=quant_mode, mxfp4_transport=mxfp4_transport,
        smoothquant_mode=smoothquant_mode,
    )


def _clear_stage1_compile_caches():
    from .mega_moe_stage1_main_a8w4 import compile_mega_moe_stage1 as compile_a8w4
    from .mega_moe_stage1_smooth import compile_mega_moe_stage1 as compile_smooth

    compile_a8w4.cache_clear()
    compile_smooth.cache_clear()


compile_mega_moe_stage1.cache_clear = _clear_stage1_compile_caches


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None, external_grouping=None, external_counting=None,
    payload_chunk_rows=0, payload_tile_ready=False,
    swiglu_limit=0.0,
    quant_mode="a8w4", compact_src=None, compact_experts=None, compact_weights=None,
    qscale_w=None, qzero_w=None, mxfp4_transport=False, transport_smooth=None,
    addr_quant_count=0, smoothquant_mode="none"):
    common = dict(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank,
        experts_per_rank=experts_per_rank, fuse_npes=fuse_npes,
        fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim,
        fixed_slot_dispatch=fixed_slot_dispatch, num_cu=num_cu,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k,
        num_waves=num_waves, grid_mult=grid_mult,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
        swizzle_a=swizzle_a, async_a_copy=async_a_copy,
        num_dispatch_cu=num_dispatch_cu,
        use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_nt=b_nt,
        swiglu_limit=swiglu_limit,
    )
    positional = (
        out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids,
        num_valid_ids, out_scale, tokens, addr_disp, i32_cur_tok,
        addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity,
        addr_expected, stream,
    )
    if quant_mode == "a8w4":
        if mxfp4_transport:
            raise ValueError("MXFP4 transport requires a SmoothQuant compute mode")
        if smoothquant_mode != "none":
            raise ValueError("fused SmoothQuant prepare is not an A8W4 MX-scale mode")
        _validate_a8w4_decode_contract(
            model_dim=model_dim, inter_dim=inter_dim,
            experts_per_rank=experts_per_rank, fuse_npes=fuse_npes,
            fuse_topk=fuse_topk, fuse_mtpr=fuse_mtpr,
            fuse_scale_dim=fuse_scale_dim,
            fixed_slot_dispatch=fixed_slot_dispatch,
            external_grouping=external_grouping,
            external_counting=external_counting,
            payload_chunk_rows=payload_chunk_rows,
            payload_tile_ready=payload_tile_ready,
            work_shards=work_shards,
        )
        from .mega_moe_stage1_main_a8w4 import run_mega_moe_stage1 as run_a8w4

        return run_a8w4(*positional, **common)
    if quant_mode not in ("a8w4smooth", "w8a8smooth"):
        raise ValueError(f"unsupported Stage1 quant_mode={quant_mode!r}")
    from .mega_moe_stage1_smooth import run_mega_moe_stage1 as run_smooth

    smooth_common = dict(
        common,
        work_shards=work_shards,
        external_grouping=external_grouping,
        external_counting=external_counting,
        payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
    )
    if quant_mode == "a8w4smooth":
        _validate_a8w4smooth_decode(
            smooth_common,
            mxfp4_transport=mxfp4_transport,
            smoothquant_mode=smoothquant_mode,
        )
    else:
        if smooth_common["fixed_slot_dispatch"]:
            raise ValueError("w8a8smooth prefill requires compact dispatch")

    return run_smooth(
        *positional, **smooth_common, quant_mode=quant_mode,
        compact_src=compact_src, compact_experts=compact_experts,
        compact_weights=compact_weights, qscale_w=qscale_w,
        qzero_w=qzero_w, mxfp4_transport=mxfp4_transport,
        transport_smooth=transport_smooth,
        addr_quant_count=addr_quant_count,
        smoothquant_mode=smoothquant_mode,
    )
# fmt: on
