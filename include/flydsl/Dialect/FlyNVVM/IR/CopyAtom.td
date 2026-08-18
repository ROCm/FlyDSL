// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYNVVM_COPYATOM
#define FLYNVVM_COPYATOM

include "flydsl/Dialect/FlyNVVM/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// CopyOp SM75 — PTX Warp-level Matrix Load Instruction: ldmatrix
//   ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16
//===----------------------------------------------------------------------===//

def FlyNVVM_CopyOpSM75_LdMatrix : FlyNVVM_CopyOp<"CopyOpSM75_LdMatrix", "sm75.ldmatrix", []> {
  let parameters = (ins "int32_t":$num, "bool":$trans);
  let assemblyFormat = "`<` `num` `=` $num `,` `trans` `=` $trans `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp SM80 — PTX Data Movement and Conversion Instruction: cp.async
//   cp.async.{ca,cg}.shared.global (global -> shared, asynchronous)
//===----------------------------------------------------------------------===//

def FlyNVVM_CopyOpSM80_CpAsync : FlyNVVM_CopyOp<"CopyOpSM80_CpAsync", "sm80.cp.async", []> {
  let parameters = (ins "int32_t":$bitSize);
  let assemblyFormat = "`<` $bitSize `>`"; // TODO: cache modifiers
  let genVerifyDecl = 1;
}


#endif // FLYNVVM_COPYATOM
