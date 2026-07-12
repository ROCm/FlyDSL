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
// MmaOp GFX1250 — MX-scaled WMMA wave32 (E8M0 block scale, V_WMMA_SCALE)
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaOpGFX1250_WMMAScale : FlyROCDL_StatefulMmaOp<"MmaOpGFX1250_WMMAScale", "gfx1250.wmma_scale", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc,
    // opselA/opselB are forwarded verbatim as the intrinsic's scaleAType /
    // scaleBType operands (the scale-format / lane selector into the scale
    // vector), NOT an output opsel; named opsel* only for consistency with the
    // plain WMMA atom. Valid range [0, 3].
    "int32_t":$opselA,
    "int32_t":$opselB,
    // Intrinsic attributes forwarded to V_WMMA_SCALE: modC (I16 C-operand
    // modifier) and reuseA/reuseB (I1 operand-reuse scheduler hints). Always
    // printed for clarity; default 0/false leaves them elided from the emitted
    // rocdl op.
    "int32_t":$modC,
    "bool":$reuseA,
    "bool":$reuseB,
    // MX block size (elements per shared E8M0 scale): 32 selects the block-32
    // V_WMMA_SCALE form (i32 scale operands), 16 selects the block-16
    // V_WMMA_SCALE16 form (i64 scale operands). Default 32.
    "int32_t":$blockSize
  );
  let assemblyFormat = [{
    `<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc
    `,` `opselA` `=` $opselA `,` `opselB` `=` $opselB
    `,` `modC` `=` $modC `,` `reuseA` `=` $reuseA `,` `reuseB` `=` $reuseB
    `,` `blockSize` `=` $blockSize `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   /*opselA=*/0, /*opselB=*/0, /*modC=*/0, /*reuseA=*/false,
                   /*reuseB=*/false, /*blockSize=*/32);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc,
      "int32_t":$opselA, "int32_t":$opselB), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   opselA, opselB, /*modC=*/0, /*reuseA=*/false, /*reuseB=*/false,
                   /*blockSize=*/32);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc,
      "int32_t":$opselA, "int32_t":$opselB,
      "int32_t":$modC, "bool":$reuseA, "bool":$reuseB), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   opselA, opselB, modC, reuseA, reuseB, /*blockSize=*/32);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc,
      "int32_t":$opselA, "int32_t":$opselB,
      "int32_t":$modC, "bool":$reuseA, "bool":$reuseB,
      "int32_t":$blockSize), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc,
                   opselA, opselB, modC, reuseA, reuseB, blockSize);
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
