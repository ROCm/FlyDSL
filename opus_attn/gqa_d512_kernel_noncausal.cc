// GQA flash attention D=512 kernel — non-causal variant
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "gqa_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gqa_d512_kernel(opus_gqa_kargs kargs) {}
template __global__ void gqa_d512_kernel<opus_gqa_traits<16, 32, 512, 8, false>>(opus_gqa_kargs);
#else
#include "gqa_d512_kernel_template.hpp"
template __global__ void gqa_d512_kernel<opus_gqa_traits<16, 32, 512, 8, false>>(opus_gqa_kargs);
#endif
