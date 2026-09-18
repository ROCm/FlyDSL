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

    Not `Pure`: two identical marks are two distinct events, and dropping one
    because its result is unused would silently lose a record.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload);
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

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload);
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

  let arguments = (ins Optional<I64>:$payload);
  let results = (outs);

  let assemblyFormat = [{
    (`payload` $payload^ `:` type($payload))? attr-dict
  }];
}

def FlyKtrace_RangeStartOp : FlyKtrace_Op<"range_start"> {
  let summary = "Open a token range, closed by range_end with the returned token.";
  let description = [{
    Returns the slot its own record occupies, so `range_end` can pair on it.
    Preferred over push/pop when the close site is not lexically nested -- notably
    across loop iterations, where the token rides an `scf.for` iter_arg.
  }];

  let arguments = (ins StrAttr:$event_name, Optional<I64>:$payload);
  // AnyType, not FlyKtrace_TokenType: the lowering replaces this result with the
  // plain i32 slot index, and a typed constraint would make the generated
  // cast<TypedValue<TokenType>> accessor assert once that happens. The frontend
  // still builds it as !fly_ktrace.token.
  let results = (outs AnyType:$token);

  let assemblyFormat = [{
    $event_name (`payload` $payload^ `:` type($payload))? attr-dict `->` type($token)
  }];
}

def FlyKtrace_RangeEndOp : FlyKtrace_Op<"range_end"> {
  let summary = "Close the range opened by a token.";
  let description = [{
    The token operand becomes the record's start_slot field, which is what the
    host pairs on.

    A sentinel token must not record, but this op does not express that: the
    frontend has already wrapped it in an `scf.if` on the token's companion `i1`
    before the op is built. See FlyKtrace_TokenType for why that flag is not part
    of the type.
  }];

  let arguments = (ins AnyType:$token, StrAttr:$event_name,
                       Optional<I64>:$payload);
  let results = (outs);

  let assemblyFormat = [{
    $token `,` $event_name (`payload` $payload^ `:` type($payload))? attr-dict
    `:` type($token)
  }];
}

def FlyKtrace_SentinelTokenOp : FlyKtrace_Op<"sentinel_token", [Pure]> {
  let summary = "A token that records nothing, for seeding a loop-carried range.";
  let description = [{
    Produces a token whose `range_end` must not write a record, for the idiom
    where the close precedes the open in source order:

    ```
    stage = sentinel_token("k_tile")
    for i in ...:
      range_end(stage)    // records nothing on the first pass
      stage = range_start("k_tile")
    range_end(stage)
    ```

    Recording nothing is expressed by the companion `i1` the frontend keeps
    beside the token, not by this op -- `range_end` is already wrapped in an
    `scf.if` on that flag before it is built. See FlyKtrace_TokenType. This op
    therefore only has to supply a value of the right type for the iter_arg;
    the lowering gives it UNPAIRED_SLOT so that a token which somehow did reach
    a store could not alias a real record's slot.
  }];

  let arguments = (ins);
  // AnyType for the same reason as range_start -- see that op.
  let results = (outs AnyType:$token);

  let assemblyFormat = [{ attr-dict `->` type($token) }];
}

#endif // FLYKTRACE_OPS
