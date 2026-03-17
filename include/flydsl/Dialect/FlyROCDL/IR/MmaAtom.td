#ifndef FLYROCDL_MMAATOM
#define FLYROCDL_MMAATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaAtom CDNA3
//===----------------------------------------------------------------------===//

def FlyROCDL_MmaAtomCDNA3_MFMA : FlyxROCL_MmaAtom<"MmaAtomCDNA3_MFMA", "atom.cdna3.mfma", []> {
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
// MmaAtom CDNA4
//===----------------------------------------------------------------------===//

#endif // FLYROCDL_MMAATOM
