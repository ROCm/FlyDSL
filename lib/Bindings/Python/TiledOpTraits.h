// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_BINDINGS_PYTHON_TILEDOPTRAITS_H
#define FLYDSL_BINDINGS_PYTHON_TILEDOPTRAITS_H

#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"

namespace mlir::fly {

LayoutAttr tiledCopyGetTiledThrValLayoutSrc(CopyAtomType copyAtom, LayoutAttr tiledLayoutThrVal,
                                            TileAttr tileMN);

LayoutAttr tiledCopyGetTiledThrValLayoutDst(CopyAtomType copyAtom, LayoutAttr tiledLayoutThrVal,
                                            TileAttr tileMN);

LayoutAttr tiledMmaGetTiledThrValLayout(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK,
                                        TileAttr permutationMNK, MmaOperand operandId);

IntTupleAttr tiledMmaGetTileSizeMNK(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK,
                                    TileAttr permutationMNK);

LayoutAttr tiledMmaGetThrLayoutVMNK(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK);

} // namespace mlir::fly

#endif // FLYDSL_BINDINGS_PYTHON_TILEDOPTRAITS_H
