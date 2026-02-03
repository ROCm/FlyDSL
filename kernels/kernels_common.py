from _mlir import ir
from _mlir.dialects import arith as _arith, llvm as _llvm, builtin, gpu as _gpu
from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl, llvm
from flydsl.lang.ir.types import T, memref
import torch


def get_torch_stream_as_async_token(loc=None, ip=None):
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream
    
    ptr_const = arith.i64(stream_ptr, loc=loc, ip=ip)
    stream_llvm_ptr = buffer_ops.create_llvm_ptr(ptr_const)
    
    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]