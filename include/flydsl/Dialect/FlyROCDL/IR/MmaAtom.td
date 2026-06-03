// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_MMAATOM
#define FLYROCDL_MMAATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaOp CDNA3
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpCDNA3_MFMA : FlyROCDL_MmaOp<"MmaOpCDNA3_MFMA", "cdna3.mfma", []> {
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

def FlyROCDL_MmaOpCDNA4_MFMAScale : FlyROCDL_StatefulMmaOp<"MmaOpCDNA4_MFMAScale", "cdna4.mfma_scale", []> {
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

def FlyROCDL_MmaOpGFX1250_WMMA : FlyROCDL_MmaOp<"MmaOpGFX1250_WMMA", "gfx1250.wmma", []> {
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
// MmaOp GFX11 — WMMA wave32 (RDNA3 / RDNA3.5: gfx1100, gfx1101, gfx1102,
// gfx1103, gfx1150, gfx1151, gfx1152)
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpGFX11_WMMA : FlyROCDL_MmaOp<"MmaOpGFX11_WMMA", "gfx11.wmma", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc,
    // Integer-WMMA controls, forwarded to the ROCDL iu8/iu4 intrinsic. Ignored
    // (and required to be false) on fp16/bf16 paths — those intrinsics have no
    // such operands. Always-printed in the assembly format for clarity.
    "bool":$signA,
    "bool":$signB,
    "bool":$clamp
  );
  let assemblyFormat = "`<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `,` `signA` `=` $signA `,` `signB` `=` $signB `,` `clamp` `=` $clamp `>`";

  let builders = [
    // Legacy 6-arg builder: defaults signA/signB/clamp=false. Backward-compat
    // for fp16/bf16 callers; integer callers should use the 9-arg form below.
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k, "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc, false, false, false);
    }]>,
    // Explicit 9-arg builder for integer-WMMA callers.
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k, "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc, "bool":$signA, "bool":$signB, "bool":$clamp), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc, signA, signB, clamp);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_MMAATOM
