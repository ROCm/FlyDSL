# API Stability

This document defines which `flydsl` APIs you can rely on across releases and
the compatibility commitments that apply when they change. Its scope is
limited to `python/flydsl/`. Below, `fx` refers to `flydsl.expr`.

A release means a minor-version increment (for example, `0.3` → `0.4`). A
patch release must not break a stable API.

## 1. Summary and general rules

- **stable**: backward compatible across minor releases. Previously valid call
  paths, signatures, and semantics must continue to work.
- **unstable**: everything else. These APIs may change or be removed in any
  minor release without notice.

`deprecated` is not a third stability level. It is a lifecycle marker for a
stable API: during its deprecation window, it remains stable and protected by
this document's compatibility rules; it may be removed only after that window
ends.

### Returned-object rule

An object returned by a stable API has a stable public result interface: its
non-underscore members and Python special methods are stable when reached
through that result. This rule applies recursively to objects returned by those
members.

This rule does not promise the concrete implementation class, its constructor,
or its import path. They may change if the replacement preserves the result
interface's public members, signatures, and semantics. For example,
`flyc.from_torch_tensor(x).mark_shape_dynamic(0)` is stable, while
`flyc.from_torch_tensor(x)._ensure_spec()` is unstable.

Determine stability by the mechanical process below. A public export from an
`expr` direct-child module has several equivalent access forms:
`fx.<name>`, `fx.<module>.<name>`, and
`from flydsl.expr.<module> import <name>` have the same stability. Otherwise,
classify the path that a caller actually uses, not the object's identity.

## 2. Determining stability

Apply the following branches by path prefix:

1. A module-global name or namespace segment beginning with `_` is unstable.
   On stable types and returned objects, only `__...__` Python special methods
   may begin with `_`; all other underscore-prefixed members are unstable.
2. A path under `flydsl.expr`: classify it only under §2.1; it is unstable if it
   does not qualify.
3. A path under `flydsl.compiler`: direct members in `compiler.__all__`, and
   members in `compiler.protocol.__all__`, are stable. Other deep paths are
   stable only when listed exactly in §2.3.
4. Every other path is stable only when listed exactly in §2.3.

For a member reached through a returned object, first classify its producing API
under these branches, then apply the returned-object rule in §1.

### 2.1 `flydsl.expr`

`expr/__init__.py` is the sole top-level aggregation manifest. It has no
`__all__` of its own; stability is determined exclusively by the following two
export chains.

#### Symbols exported from direct-child modules

Subject to the first branch above, `<name>` in a direct-child module `<module>`
is stable if and only if:

1. `expr/__init__.py` aggregates that direct-child module through
   `from .<module> import *`;
2. `<name>` is in that child module's `__all__`.

Names satisfying these conditions have identical stability when accessed via
`fx.<name>`, `fx.<module>.<name>`, or a direct import. Those direct-child
modules and their public export paths are part of the compatibility commitment.
Therefore, both `fx.Int32` and `fx.numeric.Int32` are stable. Members omitted
from `__all__` remain unstable; `fx.arith._to_raw` is unstable under the first
branch even if it was listed in a historical `__all__`.

#### Backend entry points and recursive child namespaces

`fx.<backend>...<name>` is stable if and only if:

1. the first-level `<backend>` appears in `_BACKEND_MODULES` in `expr/__init__.py`;
2. every following child namespace appears in the `__all__` of its direct parent
   package; and
3. the final `<name>` appears in the `__all__` of its owning module or package.

The first-level backend entry point, and intermediate child namespaces that
satisfy condition 2, are stable namespaces as well. This rule applies
recursively and has no path-depth limit.

For example, `fx.rocdl.cdna3.s_waitcnt` satisfies the complete export chain and
is stable, while `fx.rocdl.cluster.*` is unstable when `cluster` is not listed
in `rocdl.__all__`. An upstream-MLIR ODS builder that is re-exported but omitted
from the final `__all__` is unstable as well.

`__all__` is not access control: an attribute may exist and be callable while
still failing the stability test above.

### 2.2 `flydsl.compiler`

Only direct members of `flydsl.compiler.__all__` are stable. For example, if
`jit` is in that manifest, `flydsl.compiler.jit` is stable.

`flydsl.compiler.protocol` is an exception: every non-underscore
`flydsl.compiler.protocol.<name>` in its `__all__` is stable. It is the public
extension namespace for user implementations of JIT / DSL-value protocols.

This rule is otherwise not recursive:
`flydsl.compiler.<submodule>.<name>` does not become stable merely because
`<submodule>` is importable; it must be listed explicitly in §2.3.

### 2.3 Other explicitly stable APIs

The following full paths are stable. This table is the only exception list; a
new commitment must add an explicit row.

| API | Description |
|---|---|
| `flydsl.runtime.device.get_rocm_arch` | Query the target ROCm architecture |
| `flydsl.runtime.device.is_rdna_arch` | Choose between CDNA and RDNA paths |

### 2.4 All other APIs

Every API that does not satisfy §2.1 or §2.2, and is not listed exactly in
§2.3, is unstable. This includes undeclared `flydsl.*` submodules,
`flydsl._mlir.*`, and every underscore-prefixed name.

Direct invocation of an upstream MLIR dialect op is allowed but unstable: its
name, arguments, and semantics are controlled by upstream MLIR, and FlyDSL
provides no compatibility commitment for it.

### Stable API catalog

Generate the complete non-deprecated stable API catalog from the checked-out
source:

```bash
python3 scripts/list_stable_apis.py
python3 scripts/list_stable_apis.py --format json
```

The script reads export manifests without importing FlyDSL. It lists canonical
module paths, omitting equivalent top-level `flydsl.expr` aliases. Deprecated
APIs remain compatible but are intentionally excluded from this catalog.

## 3. Stable but deprecated

The APIs in the table below satisfy the stable rules in §2 but are marked as
deprecated. They remain stable during the §5 window; new code must not use
them, and you must provide a replacement before removal.

For direct-child module exports under §2.1, the deprecated status in this table
also applies to `fx.<name>`, `fx.<module>.<name>`, and direct-import forms.
FlyDSL maintains this table separately as the compatibility-debt list and
excludes it from the catalog above.

| API | Replacement or required work before removal | Declared removal release |
|---|---|---|
| `fx.get` | `fx.get_().unpack()` or `IntTuple[].unpack()` | v0.4 |
| `fx.index_cast` | `fx.Index(x)` | v0.4 |
| `fx.constant_vector` | `Numeric` and `Vector` member functions | v0.4 |
| `fx.tdm_ops` and content reached through this alias | `fx.rocdl.tdm_ops`; the latter is a target-specific unstable path | v0.4 |
| `fx.Numeric.maximumf`, `fx.Numeric.minimumf` | `fx.max(x, y)`, `fx.min(x, y)` | v0.4 |
| `fx.Numeric.shrui`, `fx.Numeric.addf` | `fx.arith.shrui(x, amount)`, `x + y` with a `fastmath` context | v0.4 |
| `fx.Numeric.exp2` | `fx.math.exp2(x)` | v0.4 |
| `fx.Numeric.shuffle_xor` | `fx.gpu.shuffle_xor(x, offset, width)` | v0.4 |
| `fx.rocdl.BufferCopyLDS64b` | `fx.rocdl.BufferCopyLDS32b`, or `fx.rocdl.BufferCopyLDS128b` on gfx950. No AMD target has an 8-byte LDS DMA instruction, so this entry point could never emit a working copy; it now raises instead of silently failing instruction selection. | v0.4 |

## 4. What counts as a breaking change

For a stable API, each of the following is a breaking change:

- removing a stable API or breaking its export chain: removing an entry from a
  direct-child module's `__all__`, `flydsl.compiler.__all__`, or
  `flydsl.compiler.protocol.__all__`; or removing a §2.3 entry without a new
  rule that covers it;
- removing an argument, renaming an argument that may be passed by keyword,
  reordering positional arguments, removing a default, or changing a default;
- narrowing accepted types, architectures, or value ranges;
- changing a returned scalar or value type, tuple arity, or a returned object's
  stable public result interface (including non-private members and `__...__`
  Python special methods), or the numerical, layout, or emitted-op semantics
  for previously valid input.

The following are not breaking changes:

- adding an API, or adding an optional keyword argument whose default preserves
  existing behavior;
- widening accepted input;
- improving an error message, converting undefined behavior into a clear error,
  or changing the exception type on failure;
- replacing a returned object's concrete implementation class while preserving
  its stable public result interface;
- changing only unstable APIs.

## 5. Changing or retiring a stable API

1. Provide a replacement first, normally exposing it from an appropriate stable
   path.
2. Mark the original API as deprecated at its definition or export declaration,
   and add the API and replacement to §3.
3. Retain the old API in the release where it is marked, `N`, and in the next
   minor release, `N+1`.
4. Remove the old API no earlier than `N+2`, and remove its corresponding §3
   entry at the same time.
