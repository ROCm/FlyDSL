// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYKTRACE_TYPEDEFS
#define FLYKTRACE_TYPEDEFS

include "mlir/IR/AttrTypeBase.td"

include "flydsl/Dialect/FlyKtrace/IR/Dialect.td"

class FlyKtrace_TypeDef<string name, string mnemonic_, list<Trait> traits = []>
    : TypeDef<FlyKtrace_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def FlyKtrace_TokenType : FlyKtrace_TypeDef<"Token", "token"> {
  let summary = "Buffer slot of a range's START record, for its END to pair on.";
  let description = [{
    Produced by `range_start`, consumed by `range_end`. It stands for the absolute
    slot the START record occupies -- ranges pair by slot rather than by name,
    because N iterations of one loop share a name but must not share a pairing.

    The sentinel half of the pair is NOT carried here. A token that records
    nothing also has an `i1` beside it, and `range_end` is already wrapped in an
    `scf.if` on that flag by the frontend, before the op exists. The flag cannot
    be folded into this type: across an `scf.for` that merges a sentinel
    initialiser with an in-body `range_start` it is a genuine loop-carried block
    argument, not a compile-time constant.

    The ops that produce and consume it take `AnyType`, so this type is a
    convention the frontend follows rather than a verified constraint: the
    lowering rewrites the value to a plain `i32` slot index in place, which a
    typed accessor would reject.
  }];
}

#endif // FLYKTRACE_TYPEDEFS
