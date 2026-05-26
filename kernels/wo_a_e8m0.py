"""Runtime launcher for DeepSeek-V4 grouped output projection (wo_a) with fp8 weights.

The wo_a weight is stored as fp8_e4m3fn — a self-describing floating-point type.
No external scale tensor is needed. V_CVT_SCALE_PK8_BF16_FP8 with E8M0=127 (=1.0)
is a plain fp8→bf16 type conversion.

Runtime:  launch_wo_a_bmm — call bmm_a16w8 kernel (no_scale=True) on flat-token activations
"""

import math
import torch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250

_fn_cache: dict = {}
_dummy_scale = torch.zeros(1, dtype=torch.uint8)


def launch_wo_a_bmm(
    o: torch.Tensor,
    weight_gdr: torch.Tensor,
    stream=None,
) -> torch.Tensor:
    """Run fp8 BMM for grouped output projection. No scale — fp8 values are native floats.

    Args:
        o:          [T, G, D]  bfloat16  (flat tokens × groups × d_per_group)
        weight_gdr: [G, D, R]  float8_e4m3fn  (BMM B layout: batch-first, K=D, N=R)

    Returns:
        out:        [T, G, R]  bfloat16
    """
    T, G, D = o.shape
    R = weight_gdr.shape[2]

    tile_m = 64 if T <= 128 else 128
    tile_m_pad = math.ceil(T / tile_m) * tile_m

    # cluster_n: gy = R//tile_n N-tile WGs per (batch, M-tile) all read the same A tile.
    # With cluster_n=gy, 1 WG loads A and multicasts to all others → A HBM reads ÷ gy.
    # For V4: R=1024, tile_n=128 → gy=8, cluster_n=8 ≤ 16 ✓, 1 cluster per (bz, bx).
    _gy = R // 128  # tile_n is fixed at 128
    cluster_n = _gy if 1 < _gy <= 16 else 1

    key = (G, tile_m_pad, D, R, tile_m, cluster_n)
    if key not in _fn_cache:
        _fn_cache[key] = compile_bmm_a16w8_gfx1250(
            B=G, M=tile_m_pad, N=R, K=D,
            group_k=128, group_n=128,
            tile_m=tile_m, tile_n=128, tile_k=128,
            num_buffers=3,
            use_tdm_store=True,
            no_scale=True,
            cluster_n=cluster_n,
        )

    # BMM kernel expects A: [G, T, D]
    o_bmm = o.permute(1, 0, 2).contiguous()
    if tile_m_pad > T:
        a = o_bmm.new_zeros(G, tile_m_pad, D)
        a[:, :T, :] = o_bmm
    else:
        a = o_bmm

    c = torch.zeros(G, tile_m_pad, R, dtype=torch.bfloat16, device=o.device)

    if stream is None:
        stream = torch.cuda.current_stream()

    dummy = _dummy_scale.to(o.device)
    _fn_cache[key](c.view(-1), a.view(-1), weight_gdr.view(-1),
                   dummy, tile_m_pad, stream)

    # [G, T, R] → [T, G, R]
    return c[:, :T, :].permute(1, 0, 2).contiguous()
