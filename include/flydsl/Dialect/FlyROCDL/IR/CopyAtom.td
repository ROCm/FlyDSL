#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOp_BufferLSA : FlyxROCL_CopyOp<"CopyOp_CDNA3_BufferLSA", "op.cdna3.buffer_lsa",[]> {}

#endif // FLYROCDL_COPYATOM
