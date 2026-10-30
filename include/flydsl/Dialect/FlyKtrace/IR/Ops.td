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
    Expands to one timestamped record per wave, in the next slot of the wave's
    region. Carries no result: a point event is never paired with anything.

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
    $event_name (`payload` $payload^ `:` type($payload))? attr-dict
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
    $event_name (`payload` $payload^ `:` type($payload))? attr-dict
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
    (`payload` $payload^ `:` type($payload))? attr-dict
  }];
}

def FlyKtrace_RangeStartOp : FlyKtrace_Op<"range_start"> {
  let summary = "Open a token range, closed by range_stop with the returned token.";
  let description = [{
    Returns the slot its own record occupies, so `range_stop` can pair on it.
    Preferred over push/pop when the close site is not lexically nested -- notably
    across loop iterations, where the token rides an `scf.for` iter_arg.

    The slot is wave-uniform: every active lane holds it, so the paired `range_stop`
    pairs whichever of those lanes reach it.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs AnyType:$token);

  let assemblyFormat = [{
    $event_name (`payload` $payload^ `:` type($payload))? attr-dict `->` type($token)
  }];
}

def FlyKtrace_RangeStopOp : FlyKtrace_Op<"range_stop"> {
  let summary = "Close the range opened by a token.";
  let description = [{
    The token operand becomes the record's start_slot field, which is what the
    host pairs on. The lanes that reach a stop can hold different tokens, as after
    a loop they leave at different iterations: the stop closes the latest range
    any of them started, which is the highest slot they hold.

    An inactive token is 0, below every slot, and a stop whose lanes all hold it
    records nothing: the lowering compares the slot against 0 instead of
    branching, so a loop-carried token adds no control flow to the loop.
  }];

  let arguments = (ins AnyType:$token, StrAttr:$event_name,
                       Optional<I64>:$payload,
                       DefaultValuedAttr<StrAttr, "\"uint64\"">:$payload_type);
  let results = (outs);

  let assemblyFormat = [{
    $token `,` $event_name (`payload` $payload^ `:` type($payload))? attr-dict
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

    The lowering gives it 0, which no record's slot can take, and `range_stop`
    writes nothing and takes no slot when every lane that reaches it holds that
    value.
  }];

  let arguments = (ins);
  let results = (outs AnyType:$token);

  let assemblyFormat = [{ attr-dict `->` type($token) }];
}

#endif
