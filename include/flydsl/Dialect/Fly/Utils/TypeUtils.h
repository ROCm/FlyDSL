// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_TYPEUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_TYPEUTILS_H

#include "mlir/IR/Types.h"

namespace mlir::fly {

/// Return the SSA value type that loading the storage element type \p elemTy produces.
///
/// FlyDSL keeps the *logical* signedness of an integer dtype in the element type of a
/// storage descriptor (`!fly.memref<ui8, ...>`, `!fly.ptr<ui8, ...>`), because a signless
/// `i8` cannot say whether the bytes behind it are `uint8` or `int8`. SSA values, on the
/// other hand, are always signless: the `arith` dialect only accepts signless integers and
/// the signedness of an operation is carried by the opcode (`divui` vs `divsi`). This maps
/// a scalar storage element type to the value type that loading it produces; the Python
/// side additionally unwraps vectors.
///
/// Signedness is spelled on storage only where the signless type would be ambiguous, so
/// the DSL emits `uiN` and plain `iN`, never `siN`. Accordingly only `uiN` is mapped here;
/// everything else, `siN` included, is returned unchanged so that an unexpected spelling
/// fails verification instead of being silently accepted. The Python
/// `_ssa_value_ir_type` in `expr/numeric.py` is the same function on the DSL side.
Type toSSAValueType(Type elemTy);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_TYPEUTILS_H
