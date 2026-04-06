'''
def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x(M,N): The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4(M, N//2), blockscale_e8m0((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M))).
    """
'''

#MXFP4_QUANT_BLOCK_SIZE = 32
#calculate scales across columns of x, group 32 cols(MXFP4_QUANT_BLOCK_SIZE)

#per thread load 1xMXFP4_QUANT_BLOCK_SIZE
#calculate scale
    #take abs value of all values of x
    #max value across each block of MXFP4_QUANT_BLOCK_SIZE
    #convert to int32
    #do add conversion to uint32 and "and"
    #convert to float32
    #take log2
    #clamp the value between -127 an 127
    #convert to uint8 and add 127
    #take exp2 of negatie value

#compute quantized x(multiply each x value with above calculated scale)

#convert qx to uint32

#extract sign
#set all values in quantized x to positive


import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl import _mlir
from flydsl._mlir.dialects import arith, math
from flydsl.expr.typing import T, _vec

#TODO: Set this based on device
#Wavefront size for MI3XX or lower is 64. MI4XX is 32
WAVEFRONT_SIZE = 64 #Hardcoded for MI3XX or lower
VEC_WIDTH = 8

@flyc.kernel
def _dynamic_mxfp4_quant_kernel(
    x: fx.Tensor,
    x_fp4: fx.Tensor,
    blockscale_e8m0: fx.Tensor,
    BLOCK_M: fx.Constexpr[int],
    MXFP4_QUANT_BLOCK_SIZE: fx.Constexpr[int]
):

    bidx = fx.block_idx.x
    bidy = fx.block_idx.y
    tidx = fx.thread_idx.x

    fx.printf("[kernel] bid(x,y)={},{} tid(x)={}", bidx, bidy, tidx)

    #fx.printf("[kernel] bid(x)={} tid(x)={}", bid, tid)

    x = fx.rocdl.make_buffer_tensor(x)
    x_fp4 = fx.rocdl.make_buffer_tensor(x_fp4)

    #zipped divide to get this block's layout
    mxfp4_quant_block_size = MXFP4_QUANT_BLOCK_SIZE*1
    block_m = BLOCK_M*1
    block_tile = fx.make_tile([fx.make_layout(block_m,1), fx.make_layout(mxfp4_quant_block_size,1)])
    gX = fx.zipped_divide(x, block_tile)
    #slice to get this block's tile
    bX = fx.slice(gX, ((None, None), (bidx,bidy)))
    gXfp4 = fx.zipped_divide(x_fp4, block_tile)
    bXfp4 = fx.slice(gXfp4, ((None, None), (bidx,bidy)))
    
    thr_layout = fx.make_layout((64, 1), (1, 1))
    val_layout = fx.make_layout((1, 32), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    layout_thr_val = fx.raked_product(thr_layout, val_layout)
    #fx.printf("layout_thr_val={}", layout_thr_val)
    thread_tiler = fx.make_tile(64, 32)

    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, thread_tiler)
    #fx.printf("tiled_copy={}", tiled_copy)
    thr_copy = tiled_copy.get_slice(tidx)

    
    partition_src = thr_copy.partition_S(bX)
    partition_dst = thr_copy.partition_D(bXfp4)

    frag = fx.make_fragment_like(partition_src)

    fx.copy(copy_atom, partition_src, frag)

    v = frag.load()
    vec_i16_ty = T.vec(32, T.i16)
    v = fx.vector.bitcast(vec_i16_ty, v)
    v_abs = v & 0x7FFF #Abs value. Remove sign bit of float type
    amax = fx.vector.reduction(T.i16, "maxsi", v_abs)
    vec_i32_ty = T.vec(1, T.i32)
    amax = fx.arith.extsi(T.i32, amax)
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
    quant_scale = math.exp2(scale_e8m0_unbiased)


    frag.store(v)
    fx.copy(copy_atom, frag, partition_dst)
 
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
    num_threads_per_block = WAVEFRONT_SIZE * NUM_WARPS 
    block = [num_threads_per_block,1,1]
    grid = [(M + num_threads_per_block - 1)// num_threads_per_block, 1, 1]

    _dynamic_mxfp4_quant_kernel(x,x_fp4,blockscale_e8m0,num_threads_per_block, MXFP4_QUANT_BLOCK_SIZE).launch(grid=grid, block=block)


def dynamic_mxfp4_quant(
    x: torch.Tensor, #fp16 or bf16
    scaling_mode: str= "even"
):
    M, N = x.shape

    assert (N // 2) % 2 == 0

    MXFP4_QUANT_BLOCK_SIZE = 32 #Fixed for MXFP4 format
    #x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    x_fp4 = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
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
x = torch.full((M, N), -1, dtype=torch.bfloat16).cuda()
x_fp4, blockscale_e8m0 = dynamic_mxfp4_quant(x)
#print(x)
#print(x_fp4)