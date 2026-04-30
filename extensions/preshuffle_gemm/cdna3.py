# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import argparse
from typing import Type

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from tests.utils import shuffle_weight


class PreshuffleGemmCDNA3:
    def __init__(
        self,
        ab_dtype: Type[fx.Numeric],
        out_dtype: Type[fx.Numeric],
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 64,
        output_reshuffle: bool = False,
        waves_per_eu: int | None = None,
    ):
        self.ab_dtype = ab_dtype
        self.out_dtype = out_dtype

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

        self.lds_stages_a = 2

        self.output_reshuffle = output_reshuffle
        self.waves_per_eu = waves_per_eu

    def mainloop_scheduler(self):
        fx.rocdl.sched_dsrd(2)
        fx.rocdl.sched_mfma(2)
        fx.rocdl.sched_dsrd(1)
        fx.rocdl.sched_mfma(1)
        fx.rocdl.sched_dsrd(1)
        fx.rocdl.sched_mfma(2)

        def sched_main_iter(with_vmem=False, with_dswr=False):
            if with_vmem:
                fx.rocdl.sched_vmem(1)
            fx.rocdl.sched_mfma(2)
            fx.rocdl.sched_dsrd(1)
            fx.rocdl.sched_mfma(2)
            if with_dswr:
                fx.rocdl.sched_dswr(1)

        for _ in fx.range_constexpr(8):
            sched_main_iter(with_vmem=True)
        sched_main_iter()
        for _ in fx.range_constexpr(7):
            sched_main_iter(with_dswr=True)

        fx.rocdl.sched_barrier(0)

    @flyc.kernel
    def preshuffle_gemm_cdna3(
        self,
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        layout_sA: fx.ComposedLayout,
        is_k_tile_even: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        gA_k = fx.flat_divide(A, (self.tile_m, self.tile_k))[None, None, bid_x, None]  # (BM, BK, k)
        gB_k = fx.flat_divide(B, (self.tile_n, self.tile_k))[None, None, bid_y, None]  # (BN, BK, k)
        gC = fx.flat_divide(C, (self.tile_m, self.tile_n))[None, None, bid_x, bid_y]  # (BM, BN)

        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

        copy_g2r_AB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), self.ab_dtype)
        copy_s2r_A = fx.make_copy_atom(fx.UniversalCopy128b(), self.ab_dtype)
        copy_r2g_C = fx.make_copy_atom(fx.rocdl.BufferCopy(self.out_dtype.width), self.out_dtype)

        thr_copy_s2r_A = fx.make_tiled_copy_A(copy_g2r_AB, tiled_mma).get_slice(tid)
        thr_copy_g2r_B = fx.make_tiled_copy_B(copy_g2r_AB, tiled_mma).get_slice(tid)
        thr_copy_r2g_C = fx.make_tiled_copy_C(copy_r2g_C, tiled_mma).get_slice(tid)

        sA = fx.make_view(fx.recast_iter(self.ab_dtype, fx.get_dyn_shared()), layout_sA)  # (BM, BK, STAGES_A)

        thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)  # (VA, VM, VK, k)
        thr_sA = thr_copy_g2s_A.partition_D(sA)  # (VA, VM, VN, STAGES_A)

        thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)  # (VA, VM, VK, STAGES_A)
        thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, k)
        thr_gC = thr_copy_r2g_C.partition_S(gC)  # (VC, VM, VN)

        copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])  # (VA, VM, VN)

        mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])  # (VA, VM, VN)
        mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)  # (VB, VM, VK, 2)
        mma_frag_C = thr_mma.make_fragment_C(gC)  # (VC, VM, VN)

        mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
        mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

        problem_k = fx.get_scalar(A.shape[1])
        gA_k_stride = fx.get_scalar(gA_k.stride[2])
        gB_k_stride = fx.get_scalar(gB_k.stride[2])

        def run_pipeline_stage(read_stage, next_k=None):
            write_stage = read_stage ^ 1

            if fx.const_expr(next_k is not None):
                next_k = fx.Int32(next_k)
                fx.copy(
                    copy_g2r_AB.set_value("soffset", next_k * gA_k_stride),
                    thr_gA_k[None, None, None, 0],
                    copy_frag_A,
                )
                fx.copy(
                    copy_g2r_AB.set_value("soffset", next_k * gB_k_stride),
                    thr_gB_k[None, None, None, 0],
                    mma_frag_B_retile[None, None, None, write_stage],
                )

            for block_k_iter in fx.range_constexpr(self.tile_k // fx.get_scalar(tiled_mma.tile_size_mnk[2])):
                fx.copy(
                    copy_s2r_A,
                    thr_sA_s2r[None, None, block_k_iter, read_stage],
                    mma_frag_A_retile[None, None, block_k_iter],
                )
                fx.gemm(
                    tiled_mma,
                    mma_frag_C,
                    mma_frag_A[
                        None,
                        None,
                        (None, block_k_iter) if self.tile_k != fx.get_scalar(tiled_mma.tile_size_mnk[2]) else None,
                    ],
                    mma_frag_B[
                        None,
                        None,
                        (None, block_k_iter) if self.tile_k != fx.get_scalar(tiled_mma.tile_size_mnk[2]) else None,
                        read_stage,
                    ],
                    mma_frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

            fx.copy(copy_s2r_A, copy_frag_A, thr_sA[None, None, None, write_stage])
            fx.gpu.barrier()

            self.mainloop_scheduler()

        fx.copy(copy_g2r_AB, thr_gA_k[None, None, None, 0], copy_frag_A)
        fx.copy(copy_g2r_AB, thr_gB_k[None, None, None, 0], mma_frag_B_retile[None, None, None, 0])

        mma_frag_C.fill(0)

        fx.copy(copy_s2r_A, copy_frag_A, thr_sA[None, None, None, 0])
        fx.gpu.barrier()

        for k_iter in range(0, problem_k // self.tile_k - 2, 2):
            run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
            run_pipeline_stage(read_stage=1, next_k=k_iter + 2)

        if fx.const_expr(is_k_tile_even):
            run_pipeline_stage(read_stage=0, next_k=problem_k // self.tile_k - 1)
            run_pipeline_stage(read_stage=1)
        else:
            run_pipeline_stage(read_stage=0)

        mma_frag_C_out = fx.make_fragment_like(mma_frag_C, self.out_dtype)
        mma_frag_C_out.store(mma_frag_C.load().to(self.out_dtype))

        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_out)
        fx.copy(copy_r2g_C, mma_frag_C_retile, thr_gC)

    @flyc.jit
    def __call__(
        self,
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        problem_m = fx.get_scalar(C.shape[0])
        problem_n = fx.get_scalar(C.shape[1])
        problem_k = fx.get_scalar(A.shape[1])

        if fx.const_expr(self.ab_dtype.width == 32):
            mfma_thr_k = 2
        elif fx.const_expr(self.ab_dtype.width == 16):
            mfma_thr_k = 4
        elif fx.const_expr(self.ab_dtype.width == 8):
            mfma_thr_k = 8
        else:
            raise ValueError(f"Unsupported element width: {self.ab_dtype.width}")

        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, mfma_thr_k * 4, self.ab_dtype)),
            fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, fx.make_ordered_layout((mfma_thr_k, 4, 2), order=(0, 2, 1))),
        )
        block_size = fx.get_scalar(fx.size(tiled_mma.thr_layout_vmnk))

        if self.tile_k % fx.get_scalar(tiled_mma.tile_size_mnk[2]) != 0:
            raise ValueError()

        vals_16B = 128 // self.ab_dtype.width
        thrs_col = self.tile_k // vals_16B
        thrs_row = block_size // thrs_col

        tiled_copy_g2s_A = fx.make_tiled_copy_tv(
            fx.make_copy_atom(fx.UniversalCopy128b(), self.ab_dtype),
            thr_layout=fx.make_ordered_layout((thrs_row, thrs_col), order=(1, 0)),
            val_layout=fx.make_layout((1, vals_16B), (0, 1)),
        )

        layout_sA = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 7 - self.ab_dtype.log_width, 3)),
            fx.make_ordered_layout((self.tile_m, self.tile_k, self.lds_stages_a), (1, 0, 2)),
        )

        preshuffle_B = fx.make_view(
            fx.get_iter(B),
            fx.make_ordered_layout(
                (
                    (16, problem_n // 16),
                    (vals_16B, 4, problem_k // vals_16B // 4),
                ),  # (MFMA_M, REMAIN_M), (THR_K, MFMA_K/THR_K, REMAIN_K)
                order=((1, 4), (0, 2, 3)),
            ),
        )

        launch_fn = lambda is_k_tile_even: self.preshuffle_gemm_cdna3(
            A,
            preshuffle_B,
            C,
            tiled_mma,
            tiled_copy_g2s_A,
            layout_sA,
            is_k_tile_even=is_k_tile_even,
        ).launch(
            grid=(problem_m // self.tile_m, problem_n // self.tile_n, 1),
            block=(block_size, 1, 1),
            smem=self.ab_dtype.width * fx.get_scalar(fx.cosize(layout_sA)) // 8,
            stream=stream,
        )

        is_k_tile_even = (problem_k // self.tile_k) % 2 == 0
        if is_k_tile_even:
            launch_fn(True)
        else:
            launch_fn(False)


def _resolve_ab_dtype(ab_dtype: str) -> tuple[torch.dtype, Type[fx.Numeric]]:
    if ab_dtype == "fp16":
        return torch.float16, fx.Float16
    if ab_dtype == "fp8":
        if not hasattr(torch, "float8_e4m3fnuz"):
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fnuz.")
        return torch.float8_e4m3fnuz, fx.Float8E4M3FNUZ
    raise ValueError(f"Unsupported ab dtype: {ab_dtype!r}")


def _randn_cuda(shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float16:
        return torch.randn(*shape, dtype=dtype, device="cuda")
    return torch.randn(*shape, dtype=torch.float16, device="cuda").to(dtype)


def _reference_gemm(A: torch.Tensor, B: torch.Tensor, ab_dtype: torch.dtype) -> torch.Tensor:
    if ab_dtype == torch.float16:
        return (A @ B.T).to(torch.float32)
    return A.to(torch.float32) @ B.to(torch.float32).T


def _tolerance(expected: torch.Tensor, ab_dtype: torch.dtype) -> torch.Tensor:
    atol, rtol = (1e-3, 1e-3) if ab_dtype == torch.float16 else (1e-2, 1e-2)
    return atol + rtol * expected.abs()


def benchmark_us(fn, *args, num_warmup=2, num_iters=10, **kwargs):
    for _ in range(num_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn(*args, **kwargs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / num_iters


def run_preshuffle_gemm_test(m: int = 4096, n: int = 4096, k: int = 4096, ab_dtype: str = "fp16"):
    torch_ab_dtype, flydsl_ab_dtype = _resolve_ab_dtype(ab_dtype)

    A = _randn_cuda((m, k), torch_ab_dtype)
    B = _randn_cuda((n, k), torch_ab_dtype)
    C = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    preshuffle_B = shuffle_weight(B, layout=(16, 16))

    preshuffle_gemm_op = PreshuffleGemmCDNA3(ab_dtype=flydsl_ab_dtype, out_dtype=fx.Float16)

    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=16)
    tC = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)

    print(f"[flyc] Preshuffle GEMM test: ab_dtype={ab_dtype}, M={m}, N={n}, K={k}")
    preshuffle_gemm_op(tA, preshuffle_B, tC, stream=torch.cuda.current_stream())

    us = benchmark_us(
        lambda c, a, b: preshuffle_gemm_op(a, b, c, stream=torch.cuda.current_stream()),
        tC,
        tA,
        preshuffle_B,
    )
    flops = 2 * m * n * k
    tflops = flops / (us / 1e6) / 1e12
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS")

    torch.cuda.synchronize()
    expected = _reference_gemm(A, B, torch_ab_dtype)
    actual = C.to(torch.float32)
    diff = (actual - expected).abs()
    tol = _tolerance(expected, torch_ab_dtype)
    max_diff = diff.max().item()
    max_violation = (diff - tol).max().item()
    is_correct = max_violation <= 0
    print("Result correct:", is_correct)
    if not is_correct:
        flat_idx = torch.argmax(diff).item()
        row = flat_idx // n
        col = flat_idx % n
        print("Max violation:", max_violation)
        print("Max diff:", max_diff, "at", (row, col))
        print("Expected scalar:", expected[row, col].item())
        print("Got scalar:", actual[row, col].item())


def _parse_args():
    parser = argparse.ArgumentParser(description="Run CDNA3 preshuffle GEMM host test.")
    parser.add_argument("--ab-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("-M", "--m", type=int, default=4096)
    parser.add_argument("-N", "--n", type=int, default=4096)
    parser.add_argument("-K", "--k", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preshuffle_gemm_test(m=args.m, n=args.n, k=args.k, ab_dtype=args.ab_dtype)
