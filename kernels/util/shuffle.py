import torch


def preshuffle_fp8_weights_gfx1250(w: torch.Tensor) -> torch.Tensor:
    assert w.ndim == 2, f"Expected 2D (N, K) weight tensor, got {w.ndim}D"
    N, K = w.shape
    assert N % 16 == 0, f"N ({N}) must be divisible by 16 for WMMA preshuffling"
    assert K % 16 == 0, f"K ({K}) must be divisible by 16 for WMMA preshuffling"

    w = w.view(N // 16, 16, K // 16, 16)
    w = w.permute(0, 2, 1, 3).contiguous()
    w = w.view(N // 16, K * 16)

    # Tag so gemm_a8w8_blockscale's wrapper can recover logical (N, K) from the
    # (N//16, K*16) shuffled shape instead of misreading K as K*16.
    w.is_shuffled = True
    return w
