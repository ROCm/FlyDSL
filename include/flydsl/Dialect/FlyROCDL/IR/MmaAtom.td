// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_MMAATOM
#define FLYROCDL_MMAATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaOp CDNA3
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpCDNA3_MFMA : FlyROCL_MmaOp<"MmaOpCDNA3_MFMA", "cdna3.mfma", []> {
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

//===----------------------------------------------------------------------===//
// MmaOp CDNA4
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// MmaOp GFX1250 — WMMA wave32
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpGFX1250_WMMA : FlyROCL_MmaOp<"MmaOpGFX1250_WMMA", "gfx1250.wmma", []> {
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

#endif // FLYROCDL_MMAATOM
