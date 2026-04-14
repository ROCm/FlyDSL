import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, math
from flydsl.expr import arith, vector, range_constexpr
from flydsl.expr.typing import T, Int32 
from flydsl.expr.arith import ArithValue
from flydsl.expr import buffer_ops

#TODO: Set this based on device
#Wavefront size for MI3XX or lower is 64. MI4XX is 32
WARP_SIZE = 64 #Hardcoded for MI3XX or lower
VEC_WIDTH = 8
BLOCK_THREADS = 256


@flyc.kernel
def _dynamic_mxfp4_quant_kernel(
    x: fx.Tensor,
    x_fp4: fx.Tensor,
    blockscale_e8m0: fx.Tensor,
    BLOCK_M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    MXFP4_QUANT_BLOCK_SIZE: fx.Constexpr[int]
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
    x_fp4_rsrc = buffer_ops.create_buffer_resource(x_fp4, max_size=True)
    blockscale_e8m0_rsrc = buffer_ops.create_buffer_resource(blockscale_e8m0, max_size=True)

    elem_bytes_f16 = 2
    elem_bytes_i8 = 1
    #n = N * 1

    vec_dwords_f16 = (VEC_WIDTH*elem_bytes_f16) // 4 #4
    vec_dwords_i8 = (VEC_WIDTH*elem_bytes_i8) // 4   #2
    vec_dwords_i4 = 1   #1

    vec_type_bf16 = T.vec(VEC_WIDTH, T.bf16)
    vec_type_f32 = T.vec(VEC_WIDTH, T.f32)
    vec_type_i32 = T.vec(VEC_WIDTH, T.i32)
    vec_type_ui32 = T.vec(VEC_WIDTH, fx.typing.Uint32.ir_type)
    vec_type_i16 = T.vec(VEC_WIDTH, T.i16)
    vec_type_i8 = T.vec(VEC_WIDTH, T.i8)
    vec_type_ui8 = T.vec(VEC_WIDTH, fx.typing.Uint8.ir_type)
    vec_type_i32_pack_8xi8 = T.vec(vec_dwords_i8, T.i32)
    vec_type_i32_pack_32xi4 = T.vec(4*vec_dwords_i4, T.i32)
    vec_type_bool = T.vec(VEC_WIDTH, fx.typing.Boolean.ir_type)

    row_soffset_x = ArithValue(bid) * (BLOCK_M*MXFP4_QUANT_BLOCK_SIZE*2) 
    row_soffset_x_fp4 = ArithValue(bid) * (BLOCK_M*(MXFP4_QUANT_BLOCK_SIZE//2)*elem_bytes_i8) 
    row_soffset_blockscale = ArithValue(bid) * (BLOCK_M*MXFP4_QUANT_BLOCK_SIZE*elem_bytes_i8) 
    thr_bytes_x = ArithValue(tid) * (MXFP4_QUANT_BLOCK_SIZE*elem_bytes_f16)
    thr_bytes_x_fp4 = ArithValue(tid) * ((MXFP4_QUANT_BLOCK_SIZE//2)*elem_bytes_i8)
    thr_bytes_blockscale = ArithValue(tid) * ((MXFP4_QUANT_BLOCK_SIZE)*elem_bytes_i8)

    abs_mask = arith.constant_vector(0x7FFF, vec_type_i16)
    c_mask_ones = arith.constant_vector(0xFFFFFFFF, vec_type_i32)
    c_zero_f = arith.constant(0.0, type=T.f32)
    local_max = c_zero_f
    cached_vecs = []

    #loop 4 times
    for i in range_constexpr(4):
        #load 8xbf16 values
        col_bytes = ArithValue(thr_bytes_x) + (i * VEC_WIDTH* elem_bytes_f16)
        col_end =  arith.constant(32)
        is_valid = col_end <=32 
        dw_x = col_bytes.shrui(arith.constant(2, type=T.i32))
        x_raw_data = buffer_ops.buffer_load(
            x_rsrc, dw_x, vec_width=vec_dwords_f16,
            dtype=T.i32, soffset_bytes=row_soffset_x,mask=is_valid
            )
        
        #calculate scale
        #abs
        vec_f16 = vector.bitcast(vec_type_bf16, x_raw_data)
        vec_i16 = vec_f16.bitcast(vec_type_i16) 
        vec_abs_i16 = arith.andi(vec_i16, abs_mask)
        vec_abs = vec_abs_i16

        cached_vecs.append(vec_f16)
        local_max = fx.vector.reduction(T.i16, "maxsi", vec_abs)
    
    amax = fx.arith.extsi(T.i32, local_max)
    #fx.printf("amax={}", amax)
    c_200000 = fx.arith.constant(0x200000, type=T.i32)
    amax = amax + c_200000
    amax = fx.arith.bitcast(fx.typing.Uint32.ir_type, amax)
    amax = amax & 0xFF800000
    amax = fx.arith.bitcast(T.f32, amax)
    scale_e8m0_unbiased = math.log2(amax)
    scale_e8m0_unbiased = math.floor(scale_e8m0_unbiased) - 2
    c_m127 = fx.arith.constant(-127, type=T.f32)
    c_127 = fx.arith.constant(127, type=T.f32)
    scale_e8m0_unbiased = math.clampf(scale_e8m0_unbiased, c_m127, c_127)
    bs_e8m0 = fx.arith.bitcast(fx.typing.Uint32.ir_type, scale_e8m0_unbiased)
    bs_e8m0 = fx.arith.trunci(fx.typing.Uint8.ir_type, bs_e8m0)
    ci_127 = fx.arith.constant(127, type=fx.typing.Uint8.ir_type)
    bs_e8m0 = fx.arith.addi(bs_e8m0,ci_127)
    quant_scale = math.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    quant_scale_splat = fx.vector.broadcast(vec_type_f32, quant_scale)    
    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    EXP_BIAS_FP32 = fx.arith.constant(127)
    EXP_BIAS_FP4 = fx.arith.constant(1)
    EBITS_F32 = fx.arith.constant(8)
    EBITS_FP4 = fx.arith.constant(2)
    MBITS_F32 = fx.arith.constant(23)
    MBITS_FP4 = fx.arith.constant(1)

    max_normal = fx.arith.constant(6.0)
    min_normal = fx.arith.constant(1.0)

    e2m1_value_packed = []
    for i in range_constexpr(4):
        col_end =  arith.constant(32)
        is_valid = col_end <=32 

        x_vec_f16 = cached_vecs[i]
        x_f32 = x_vec_f16.extf(vec_type_f32)
        qx_fp32 = x_f32 * quant_scale_splat
        qx = fx.vector.bitcast(vec_type_ui32, qx_fp32)

        # Extract sign
        s = qx & 0x80000000

        # Set everything to positive, will add sign back at the end
        qx = qx ^ s

        saturate_mask = qx_fp32 >= fx.vector.broadcast(vec_type_f32, max_normal)
        saturate_mask_i32 = fx.arith.extui(vec_type_i32, saturate_mask)
        tmp = qx_fp32 < fx.vector.broadcast(vec_type_f32,min_normal)
        tmp = fx.arith.extui(vec_type_i32, tmp)
        denormal_mask = fx.arith.xori(saturate_mask_i32, c_mask_ones) & tmp
        normal_mask =  (saturate_mask_i32 | denormal_mask)
        normal_mask = fx.arith.xori(normal_mask, c_mask_ones)
        denormal_mask = fx.arith.trunci(vec_type_bool, denormal_mask) #Convert to boolean for use in arith.select
        normal_mask = fx.arith.trunci(vec_type_bool, normal_mask)#Convert to boolean for use in arith.select

        # Denormal numbers
        denorm_exp = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + fx.arith.constant(1)
        denorm_mask_int = denorm_exp << MBITS_F32
        denorm_mask_float = fx.vector.bitcast(vec_type_f32, fx.vector.broadcast(vec_type_i32, denorm_mask_int))
        denormal_x = qx_fp32 + denorm_mask_float
        denormal_x = fx.vector.bitcast(vec_type_ui32, denormal_x)
        denormal_x = denormal_x - fx.vector.broadcast(vec_type_i32, denorm_mask_int)
        denormal_x = fx.arith.trunci(vec_type_ui8, fx.arith.unwrap(denormal_x))

        # Normal numbers
        normal_x = qx_fp32
        mant_odd = (qx >> fx.vector.broadcast(vec_type_i32, (MBITS_F32 - MBITS_FP4))) & 1
        val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
        normal_x += fx.vector.broadcast(vec_type_f32, fx.arith.bitcast(T.f32, val_to_add))
        normal_x = fx.vector.bitcast(vec_type_ui32, normal_x) + fx.vector.broadcast(vec_type_ui32, mant_odd)
        normal_x = fx.arith.trunci(vec_type_ui8, fx.arith.unwrap(normal_x))

        # Merge results
        e2m1_value = fx.vector.broadcast(vec_type_ui8, arith.trunci(fx.typing.Uint8.ir_type, fx.arith.constant(0x7)))
        e2m1_value = fx.arith.select(normal_mask, normal_x, e2m1_value)
        e2m1_value = fx.arith.select(denormal_mask, denormal_x, e2m1_value)

        sign_lp = s >> fx.vector.broadcast(vec_type_ui32,(MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4))
        sign_lp = fx.arith.trunci(vec_type_ui8, fx.arith.unwrap(sign_lp))
    
        e2m1_value_ui8 = e2m1_value | sign_lp 
        e2m1_value_scalars = fx.vector.to_elements(e2m1_value_ui8)

        e2m1_value_packed.append((e2m1_value_scalars[0] << 4) | e2m1_value_scalars[1])
        e2m1_value_packed.append((e2m1_value_scalars[2] << 4) | e2m1_value_scalars[3])
        e2m1_value_packed.append((e2m1_value_scalars[4] << 4) | e2m1_value_scalars[5])
        e2m1_value_packed.append((e2m1_value_scalars[6] << 4) | e2m1_value_scalars[7])
        #fx.printf(len(e2m1_value_packed))

    e2m1_value_packed = fx.vector.from_elements(T.vec(16, fx.typing.Uint8.ir_type), e2m1_value_packed)
    out_packed = vector.bitcast(vec_type_i32_pack_32xi4, e2m1_value_packed)
    col_bytes = ArithValue(thr_bytes_x_fp4) 
    buffer_ops.buffer_store(
        out_packed, x_fp4_rsrc, col_bytes,
        soffset_bytes=row_soffset_x_fp4, offset_is_bytes=True
    )

@flyc.jit
def _dynamic_mxfp4_quant_host(
    x: fx.Tensor,
    x_fp4: fx.Tensor,
    blockscale_e8m0: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    MXFP4_QUANT_BLOCK_SIZE: fx.Constexpr[int],
    NUM_WARPS: fx.Constexpr[int]
):
    num_threads_per_block = WARP_SIZE * NUM_WARPS 
    block = [num_threads_per_block,1,1]
    grid = [(M + num_threads_per_block - 1)// num_threads_per_block, 1, 1]

    _dynamic_mxfp4_quant_kernel(x,x_fp4,blockscale_e8m0,num_threads_per_block, N, MXFP4_QUANT_BLOCK_SIZE).launch(grid=grid, block=block)


def dynamic_mxfp4_quant(
    x: torch.Tensor, #fp16 or bf16
    scaling_mode: str= "even"
):
    M, N = x.shape

    assert (N // 2) % 2 == 0

    MXFP4_QUANT_BLOCK_SIZE = 32 #Fixed for MXFP4 format
    x_fp4 = torch.empty((M, N // 2 ), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device
    ).T

    NUM_WARPS = 1
    _dynamic_mxfp4_quant_host(x, x_fp4, blockscale_e8m0, M, N, MXFP4_QUANT_BLOCK_SIZE, NUM_WARPS)

    return (x_fp4, blockscale_e8m0)


M = 64 
N = 32
torch.set_printoptions(threshold=4096)
x = torch.full((M, N), 2, dtype=torch.bfloat16).cuda()
#x = torch.reshape(x, (M,N))
x_fp4, blockscale_e8m0 = dynamic_mxfp4_quant(x)
print(x)
print(x_fp4)
#print(blockscale_e8m0)