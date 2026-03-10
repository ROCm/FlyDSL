#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOpBufferLDST : FlyxROCL_CopyOp<"CopyOpCDNA3BufferLDST", "cdna3.buffer_ldst",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

#endif // FLYROCDL_COPYATOM
