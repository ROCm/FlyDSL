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

def FlyROCDL_MmaOpCDNA4_MFMAScale : FlyROCL_StatefulMmaOp<"MmaOpCDNA4_MFMAScale", "cdna4.mfma_scale", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc,
    "int32_t":$opselA,
    "int32_t":$opselB
  );
  let assemblyFormat = [{
    `<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc
    `,` `opselA` `=` $opselA `,` `opselB` `=` $opselB `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   /*opselA=*/0, /*opselB=*/0);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc,
      "int32_t":$opselA, "int32_t":$opselB), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   opselA, opselB);
    }]>
  ];
  let genVerifyDecl = 1;
}

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

//===----------------------------------------------------------------------===//
// MmaOp RDNA3 — WMMA wave32 (gfx1100/gfx1150)
//
// RDNA 3 / RDNA 3.5 expose a single 16x16x16 WMMA instruction family:
//   F32 <- F16/F16, BF16/BF16
//   F16 <- F16/F16
//   BF16 <- BF16/BF16
//   I32 <- IU8/IU8, IU4/IU4
//
// Operand layout uses the legacy "WMMA256bInsts" convention (lanes 0-15
// hold all distinct K data; lanes 16-31 are populated by the LLVM AMDGPU
// codegen with duplicates of the lower half).  M=N=K=16 is the only
// supported shape; FP8/FP4/scaled/SWMMAC are gfx12+ only and rejected
// by `verify`.
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpRDNA3_WMMA : FlyROCL_MmaOp<"MmaOpRDNA3_WMMA", "rdna3.wmma", []> {
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
