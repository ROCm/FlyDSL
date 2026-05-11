// C ABI launcher for the OPUS GQA kernels, used by the Python ctypes wrapper.
#include <opus/hip_minimal.hpp>

#include "gqa_defs.h"

template<class Traits>
__global__ void gqa_d128_kernel(opus_gqa_kargs kargs);
template<class Traits>
__global__ void gqa_d512_kernel(opus_gqa_kargs kargs);

template<class Traits>
static inline void launch_gqa(const opus_gqa_kargs& kargs, dim3 grid, dim3 block) {
    if constexpr (Traits::D_TILE_SIZE == 128) {
        gqa_d128_kernel<Traits><<<grid, block>>>(kargs);
    } else {
        gqa_d512_kernel<Traits><<<grid, block>>>(kargs);
    }
}

static constexpr int OPUS_GQA_INVALID_VALUE = -1;

template<class Traits>
static inline int validate_and_launch(const opus_gqa_kargs& kargs) {
    if (kargs.D != Traits::D_TILE_SIZE) {
        return OPUS_GQA_INVALID_VALUE;
    }
    if ((kargs.N % Traits::KV_TILE_SIZE) != 0 ||
        (kargs.N / Traits::KV_TILE_SIZE) < 6) {
        return OPUS_GQA_INVALID_VALUE;
    }
    if ((kargs.N % (Traits::Q_TILE_SIZE * Traits::NUM_WARPS)) != 0) {
        return OPUS_GQA_INVALID_VALUE;
    }

    const int num_q_tiles = ceil_div(kargs.N, Traits::Q_TILE_SIZE);
    const int num_q_blocks = ceil_div(num_q_tiles, Traits::NUM_WARPS);
    dim3 grid(kargs.H, num_q_blocks, kargs.B);
    dim3 block(Traits::BLOCK_SIZE);

    launch_gqa<Traits>(kargs, grid, block);
    return static_cast<int>(hipGetLastError());
}

extern "C" int opus_gqa_forward(
    const void* q,
    const void* k,
    const void* v,
    void* o,
    int B,
    int N,
    int H,
    int H_KV,
    int D,
    int causal) {
    if (!q || !k || !v || !o || B <= 0 || N <= 0 || H <= 0 || H_KV <= 0 ||
        D <= 0 || (H % H_KV) != 0) {
        return OPUS_GQA_INVALID_VALUE;
    }

    opus_gqa_kargs kargs{};
    kargs.ptr_q = q;
    kargs.ptr_k = k;
    kargs.ptr_v = v;
    kargs.ptr_o = o;
    kargs.B = B;
    kargs.N = N;
    kargs.H = H;
    kargs.H_KV = H_KV;
    kargs.D = D;
    kargs.stride_q_b = N * H * D;
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_kv_b = N * H_KV * D;
    kargs.stride_kv_n = H_KV * D;
    kargs.stride_kv_h = D;

    int err;
    if (D == 128) {
        err = causal ? validate_and_launch<opus_gqa_traits<32, 64, 128, 8, true>>(kargs)
                     : validate_and_launch<opus_gqa_traits<32, 64, 128, 8, false>>(kargs);
    } else if (D == 512) {
        err = causal ? validate_and_launch<opus_gqa_traits<16, 32, 512, 8, true>>(kargs)
                     : validate_and_launch<opus_gqa_traits<16, 32, 512, 8, false>>(kargs);
    } else {
        err = OPUS_GQA_INVALID_VALUE;
    }

    return err;
}

extern "C" const char* opus_gqa_hip_error_string(int error_code) {
    if (error_code == OPUS_GQA_INVALID_VALUE) {
        return "invalid OPUS GQA launch parameters";
    }
    return hipGetErrorString(static_cast<hipError_t>(error_code));
}
