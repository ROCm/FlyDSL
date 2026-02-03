from _mlir import ir
from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl, llvm
from flydsl.lang.ir.types import T, memref
import torch

def get_torch_stream_as_mlir_value(loc=None, ip=None):
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream
    
    i64_type = ir.IntegerType.get_signless(64)
    ptr_const = arith.constant(ir.IntegerAttr.get(i64_type, stream_ptr), loc=loc, ip=ip)
    
    ptr_type = llvm.LLVMPointerType.get()
    stream_value = llvm.inttoptr(ptr_type, ptr_const, loc=loc, ip=ip)
    
    return stream_value