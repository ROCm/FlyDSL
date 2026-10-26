// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYKTRACE_OPS
#define FLYKTRACE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "flydsl/Dialect/FlyKtrace/IR/Dialect.td"
include "flydsl/Dialect/FlyKtrace/IR/TypeDefs.td"

class FlyKtrace_Op<string mnemonic, list<Trait> traits = []>
    : Op<FlyKtrace_Dialect, mnemonic, traits>;

def FlyKtrace_MarkOp : FlyKtrace_Op<"mark"> {
  let summary = "Record one point event.";
  let description = [{
    Expands to a timestamped record written by the first active lane of the wave.
    Carries no result: a point event is never paired with anything.

    `event_name` is the user-facing phase name. It stays a string here and is
    resolved to the 24-bit id the device writes when the host reads the trace
    back, so this op does not depend on the name table.

    `payload` is an optional wave-uniform scalar stored alongside the timestamp.
    Under divergence the first active lane's value is the one recorded.
    `payload_type` describes the encoded i64 operand: uint64 (the default),
    int64, float32, float64, or bool. Floating-point operands contain IEEE bits;
    signed integer operands are sign-extended before encoding.

    Not `Pure`: two identical marks are two distinct events, and dropping one
    because its result is unused would silently lose a record.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs);

  let assemblyFormat = [{
    $event_name (`payload` $payload^ `:` type($payload))? prop-dict attr-dict
  }];
}

def FlyKtrace_RangePushOp : FlyKtrace_Op<"range_push"> {
  let summary = "Open a stack range, closed by the next range_pop.";
  let description = [{
    Same record shape as `mark`, with a different kind byte. Push and pop pair
    LIFO along an executed path; the pairing is reconstructed host-side from the
    record order, so neither op carries a handle.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs);

  let assemblyFormat = [{
    $event_name (`payload` $payload^ `:` type($payload))? prop-dict attr-dict
  }];
}

def FlyKtrace_RangePopOp : FlyKtrace_Op<"range_pop"> {
  let summary = "Close the innermost open range_push.";
  let description = [{
    Carries no event name: the host takes the name from the matching push, so
    this record's event_id field is 0.
  }];

  let arguments = (ins Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs);

  let assemblyFormat = [{
    (`payload` $payload^ `:` type($payload))? prop-dict attr-dict
  }];
}

def FlyKtrace_RangeStartOp : FlyKtrace_Op<"range_start"> {
  let summary = "Open a token range, closed by range_stop with the returned token.";
  let description = [{
    Returns the slot its own record occupies, so `range_stop` can pair on it.
    Preferred over push/pop when the close site is not lexically nested -- notably
    across loop iterations, where the token rides an `scf.for` iter_arg.

    The paired `range_stop` must execute under the same guard. Only the first active
    lane holds a real slot; the rest carry UNPAIRED_SLOT, so if the two endpoints run
    with different lanes active the end has nothing to pair on and the range is
    dropped.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs AnyType:$token);

  let assemblyFormat = [{
    $event_name (`payload` $payload^ `:` type($payload))? prop-dict attr-dict `->` type($token)
  }];
}

def FlyKtrace_RangeStopOp : FlyKtrace_Op<"range_stop"> {
  let summary = "Close the range opened by a token.";
  let description = [{
    The token operand becomes the record's start_slot field, which is what the
    host pairs on. It must execute under the same guard as its `range_start` --
    see that op.

    An inactive token must not record, but this op does not express that: the
    frontend has already wrapped it in an `scf.if` on the token's companion `i1`
    before the op is built. See FlyKtrace_TokenType for why that flag is not part
    of the type.
  }];

  let arguments = (ins AnyType:$token, StrAttr:$event_name,
                       Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs);

  let assemblyFormat = [{
    $token `,` $event_name (`payload` $payload^ `:` type($payload))? prop-dict attr-dict
    `:` type($token)
  }];
}

def FlyKtrace_InactiveTokenOp : FlyKtrace_Op<"inactive_token", [Pure]> {
  let summary = "A token that records nothing, for seeding a loop-carried range.";
  let description = [{
    Produces a token whose `range_stop` must not write a record, for the idiom
    where the close precedes the open in source order:

    ```
    stage = inactive_token("k_tile")
    for i in ...:
      range_stop(stage)
      stage = range_start("k_tile")
    range_stop(stage)
    ```

    Recording nothing is expressed by the companion `i1` the frontend keeps
    beside the token, not by this op -- `range_stop` is already wrapped in an
    `scf.if` on that flag before it is built. See FlyKtrace_TokenType. This op
    therefore only has to supply a value of the right type for the iter_arg;
    the lowering gives it UNPAIRED_SLOT so that a token which somehow did reach
    a store could not alias a real record's slot.
  }];

  let arguments = (ins);
  let results = (outs AnyType:$token);

  let assemblyFormat = [{ attr-dict `->` type($token) }];
}

#endif
