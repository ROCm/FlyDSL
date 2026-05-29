import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import (
    ceildiv,
    divmod,
)


def compile_bf16_gemm(
    *,
    K: int
):
    # 256x256x64
    BLOCK_SIZE = 256
    BLOCK_K = 64

    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    BYTES_THREAD = 16
    NUM_WAVES = 8
    BYTES_WAVE = BYTES_THREAD * 64
    LDS_TILE_SIZE = BLOCK_SIZE * BLOCK_K * 2

    N_LDS_STEPS = LDS_TILE_SIZE // (BYTES_THREAD * 512)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # MFMA atom is 32x32x16
        #   - each lane holds 8 BF16 values of A/B
        #   - each lane holds 16 F32 values of C (4 row groups, each row group has 8 rows in it with each lane holding 4 consecutive elements in the column)
        # The accumulator is 128x64 -> I need 8 of these 32x32 fragments (4x2 config)
        Accum_t = Vec.make_type(16, fx.Float32)
        Accum_zero = Vec.filled(16, 0.0, fx.Float32)

        C_rsrc = buffer_ops.create_buffer_resource(C)

        c_frag = [
            [Accum_zero for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            [lds_alloc.allocate(fx.Array[fx.BFloat16, BLOCK_SIZE * BLOCK_K, 16]).peek().ptr]
            for _ in range_constexpr(2)
        ]
        B_lds = [
            [lds_alloc.allocate(fx.Array[fx.BFloat16, BLOCK_SIZE * BLOCK_K, 16]).peek().ptr]
            for _ in range_constexpr(2)
        ]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        # This tile (C) row, col
        row, col = divmod(fx.block_idx.x, ceildiv(c_n, BLOCK_SIZE))
        # The row, col of the sub-tile for this wave
        wave_row, wave_col = divmod(wave_id, 4)

        def _load_lds(gl_src, lds_dst, k_offset):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            for step in range_constexpr(N_LDS_STEPS):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + fx.Int32(wave_id * BYTES_WAVE + step * BYTES_WAVE * NUM_WAVES), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    lds_ptr,
                    fx.Int32(BYTES_THREAD),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def _store_rt():
            base_row = row * BLOCK_SIZE + wave_row * (BLOCK_SIZE // 2)
            base_col = col * BLOCK_SIZE + wave_col * (BLOCK_SIZE // 4)

            for i in range_constexpr(4):
                row_offset = i * 32
                for j in range_constexpr(2):
                    col_offset = j * 32
                    v_bf16 = c_frag[i][j].to(fx.BFloat16)
                    for group_idx in range_constexpr(4):
                        r = base_row + row_offset + group_idx * 8 + (lane_id // 32) * 4
                        c = base_col + col_offset + lane_id % 32
                        for el_idx in range_constexpr(4):
                            buffer_ops.buffer_store(
                                v_bf16[group_idx * 4 + el_idx], C_rsrc, fx.Int32((r + el_idx) * c_n + c)
                            )

        cur, next = 0, 1
        for k in range_constexpr(K_ITERS - 1):
            # TODO: impl...
            cur ^= 1
            next ^= 1

        _store_rt()

    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_SIZE) * ceildiv(c_n, BLOCK_SIZE)
        kernel_gemm(
            A,
            B_T,
            C,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
