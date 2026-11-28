// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYNVVM_MMAATOM
#define FLYNVVM_MMAATOM

include "flydsl/Dialect/FlyNVVM/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaOp SM80
//   ptx: mma.sync.aligned
//===----------------------------------------------------------------------===//

def FlyNVVM_MmaOpSM80_MmaSync : FlyNVVM_MmaOp<"MmaOpSM80_MmaSync", "sm80.mma.sync", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc
  );
  let assemblyFormat = "`<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k, "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYNVVM_MMAATOM
