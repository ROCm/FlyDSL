# Debugging FlyDSL Kernels

A fair question before you commit to a new toolchain: **when a FlyDSL kernel goes
wrong, can you actually debug it — or are you staring at an opaque wall of MLIR?**

The honest answer is *yes, you can debug it*, but "debugging" here is really three
different activities with sharply different ergonomics. Trace-time and run-time
bugs are pleasant — you get ordinary Python tracebacks and `printf`. Compile-time
failures are the sharp edge: they often surface as a **C++ assertion abort** with
no Python context. This chapter maps the three layers, the tools for each, a
repeatable isolation loop, and two real failures worked end to end.

## The three layers of failure

A `.py` kernel passes through three regimes (Chapter 2). A bug shows up in
exactly one of them, and *where* it shows up dictates *how* you debug it.

| Layer | When it runs | Typical symptom | Primary tools |
|-------|--------------|-----------------|---------------|
| **Trace-time** | Python builds the IR (host) | normal Python exception, traceback at the `fx.*` line | read the traceback; `pdb`; print layouts |
| **Compile-time** | C++ pass pipeline lowers the IR (host) | **C++ assertion → `SIGABRT`** that kills the process (and any `pytest` run) | `FLYDSL_DUMP_IR`, `FLYDSL_DEBUG_PRINT_AFTER_ALL`, isolate + bisect, `fly-opt` |
| **Run-time** | kernel executes on the GPU | wrong numbers, `NaN`/`inf`, or an illegal-address fault | `fx.printf`, host comparison, all-ones test, shrink the launch |

The rule of thumb: **the deeper the layer, the less the error message tells you.**
A trace-time error names your line; a compile-time abort names a line *in the
compiler*; a run-time bug names nothing at all.

## Trace-time errors (the friendly ones)

Tracing is just Python executing your kernel body to emit IR (§2.4). So most
mistakes — a wrong-rank coordinate, a type mismatch, a protocol violation — raise
an ordinary `TypeError`/`ValueError` whose traceback points straight at the
offending `fx.*` call. Everything you already do in Python works: drop a
`breakpoint()`, `print(layout)` to see a `Shape:Stride`, or inspect a value's
type before it flows into the next op.

These are the cheapest bugs to fix: read the message, open the named line, done.
Section 8.7 works one that looks scarier than it is.

## Compile-time aborts (the hard ones)

Once your Python returns, the module enters the C++ pass pipeline. Bugs here
frequently manifest as an **LLVM/MLIR assertion failure** — a raw abort with a
C++ `file:line` and a terse message, for example:

```
IntTupleUtils.h:943 ... Assertion `!collector.empty() && "not support empty IntTuple"' failed.
Casting.h:566     ... cast<TypedValue<LLVMPointerType>>() argument of incompatible type!
```

Two properties make these unpleasant:

1. **No Python context.** The traceback into your kernel is gone; you get only
   the C++ assertion.
2. **It aborts the whole process.** One malformed kernel takes down an entire
   `pytest` session, so you cannot tell which of 15 tests died until you isolate.

Localize the failing pass with IR dumps:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/ir FLYDSL_RUNTIME_ENABLE_CACHE=0 python repro.py
ls /tmp/ir        # numbered .mlir, one per stage, plus final_isa.s
```

The **last** file written is the last stage that *succeeded* — the crash is in
the pass immediately after it. `FLYDSL_DEBUG_PRINT_AFTER_ALL=1` prints the IR
after every pass so you can watch exactly which one dies, and `fly-opt` (in
`tools/`) can replay a single pass on a dumped `.mlir` to minimize at the IR
level.

> **Tell-tale.** If the abort happens *before* any IR is dumped, it fired during
> tracing/verification — it is really a malformed-IR bug from the DSL call you
> just wrote. The assertion message names the operation (`intTupleSlice`,
> `crd2idx`, …), which points straight back at the `fx.*` you should suspect.

## Run-time bugs: wrong numbers and `NaN`

The kernel compiles and launches, but the output is wrong. You are now debugging
GPU behavior, so the tools are the classic ones:

- **`fx.printf`** — `fx.printf("bid={} tid={} v={}", bid, tid, v)`. *Guard it*
  (`if bid == 0 and tid == 0:`) or you will drown in thousands of lines.
- **Compare to a reference** — diff elementwise against a torch reference (the
  puzzles' `common.check`) and print the first mismatching index.
- **All-ones test** — feed inputs of all `1.0` so the expected output is analytic
  (a row-sum becomes `N`, a dot product becomes `K`). This separates *indexing*
  bugs from *arithmetic* bugs instantly.
- **Shrink the launch** — run one block, then one thread, then one output tile,
  and verify a single element by hand.
- **Illegal-address faults** (HIP "memory access fault") almost always mean an
  out-of-bounds gather. Re-check the offset arithmetic, and prefer the
  OOB-checked buffer path (`make_buffer_tensor` + `BufferCopy*`) or a `pred=`
  mask on `fx.copy` (§5).

> **Cache caveat (§2.7).** While iterating, always run with
> `FLYDSL_RUNTIME_ENABLE_CACHE=0`. The disk cache invalidates on kernel *source*
> and *closure* changes, but not on edits to helper code outside the traced
> closure — a stale artifact will happily hide your fix.

## The micro-repro loop

The single most effective technique — indispensable for the §8.3 aborts — is a
tight isolation loop:

1. **Extract** the failing kernel into a ~30-line standalone script, *outside*
   `pytest`, so the `SIGABRT` cannot nuke the whole suite.
2. **Shrink** the shapes to the smallest that still fails.
3. **Bisect the body.** Comment out the back half of the kernel; does it still
   abort? Narrow until a single op is responsible.
4. **Contrast** a working form against the failing one — 2-D vs 3-D slice;
   `make_view` on a raw tensor vs on `make_buffer_tensor`; `UniversalCopy` vs
   `BufferCopy`. The one-line diff that flips *pass ↔ abort* is your bug.
5. **Re-fold** the fix into the real kernel and re-run the suite.

## Worked failure #1 — the empty-`IntTuple` abort

*A compile-time abort (§8.3).* A `pytest` run died mid-suite with a `SIGABRT`:

```
IntTupleUtils.h:943 ... Assertion `... "not support empty IntTuple"' failed.
```

Running the one suspect kernel standalone reproduced it at trace time, on:

```python
fx.copy(copy, fx.slice(Xv, (c, ho + r, wo + s)), rX)   # all coords concrete
```

Hypothesis: a **fully-concrete slice** (no `None`) collapses the coordinate to an
empty tuple, which the slicer rejects. A minimal repro confirmed it — slicing a
2-D view with `(1, 2)` aborts, while `(1, None)` compiles. The fix is to **never
fully-slice; always keep ≥ 1 mode**, then re-tile it down to the single element:

```python
row  = fx.slice(view, (i, j, None))              # keep the last mode
row1 = fx.logical_divide(row, fx.make_layout(1, 1))
r    = fx.make_rmem_tensor(1, fx.Float32)
fx.copy_atom_call(copy, fx.slice(row1, (None, last)), r)
val  = fx.Vector(fx.memref_load_vec(r))[0]
```

(While isolating this, a *sibling* abort appeared —
`cast<TypedValue<LLVMPointerType>>`— from feeding a `UniversalCopy` a
`make_buffer_tensor` iterator. `UniversalCopy` needs a *plain* pointer:
`fx.make_view(fx.get_iter(T), layout)` on the **raw** tensor. Two different C++
asserts, two different DSL mistakes — both found by the same contrast method.)

## Worked failure #2 — the Storable-protocol error

*A trace-time error (§8.2) that reads worse than it is.* Allocating LDS failed at
`.allocate(SharedStorage)` with:

```
TypeError: ... field 's_red' has type 'fx.Array[fx.Float32, RED_SLOTS]'
which does not implement the Storable protocol.
```

The tell is subtle: the field type is quoted as a **string**. That only happens
under `from __future__ import annotations` (PEP 563), which stringizes every
annotation — and `@fx.struct` reads the raw `__annotations__` without resolving
them, so it tries to size the *string* instead of the type. Isolation was one
line: define the same struct with, then without, the future import — it fails
only with it. The fix: **do not put `from __future__ import annotations` in a
file that defines an `@fx.struct`** (kernel-signature annotations tolerate it;
`@fx.struct` does not).

## rocgdb and Python-line stepping

If `fx.printf` and host-side bisection are not enough, `rocgdb` — AMD's GDB port
for GPU code — can step through a running FlyDSL kernel at the Python source-line
level. Two support scripts in `scripts/` make this practical.

### What the debug info actually is

`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` does three concrete things when the kernel
is compiled:

1. Preserves Python source locations (`FileLineColLoc`, recorded at trace time
   by `capture_user_location()` in `expr/meta.py`) through the MLIR module
   re-parse boundary.
2. Appends `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}`
   to the pass pipeline — this emits DWARF **line tables only** (`.debug_line`
   section): ISA-address → Python-file:line:col, but no variable names or types.
3. Passes `-g` to the LLVM AMDGPU backend so those line tables reach the HSACO.

The source locations point back to your **Python `fx.*` call sites**, not to
MLIR ops or ISA addresses. The DWARF actually present in the binary was verified
with `scripts/flydsl_dwarf_mapper.py` on a real gfx942 kernel:

```
Line   ISA start            ISA end              #entries  File
----------------------------------------------------------------------
  37   0x0000000000001608   0x00000000000016ac   3         test_kernel.py
  40   0x0000000000001610   0x0000000000001610   1         test_kernel.py
  48   0x0000000000001618   0x0000000000001618   1         test_kernel.py
  52   0x000000000000162c   0x0000000000001734   2         test_kernel.py
  53   0x0000000000001740   0x0000000000001740   1         test_kernel.py
```

Each Python source line maps to a range of ISA addresses. rocgdb reads this
table and can therefore break *by Python line* and step *by Python line*.

**Switching to `emission-kind=Full` adds nothing.** The investigation confirmed
that `emission-kind=Full` changes only the DWARF flag — the MLIR pipeline
emits no `llvm.dbg.value`/`llvm.dbg.declare` intrinsics, so there are no
variable entries to describe regardless of the emission kind.

### The rocgdb session workflow

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  /opt/rocm/bin/rocgdb -x scripts/flydsl_gdb.py --args python3 repro.py
```

```gdb
(gdb) set breakpoint pending on
(gdb) break my_kernel_0        # symbol = <func_name>_<sequential_id>
(gdb) break repro.py:52        # OR: break on a specific Python source line
(gdb) run

# GPU wavefronts are now visible; one is halted at your breakpoint.
(gdb) flydsl-where             # show current Python line + context
(gdb) flydsl-step              # advance to next Python line (steps ISA internally)
(gdb) flydsl-regs              # v0-v15, s0-s15, EXEC mask
(gdb) flydsl-line-table        # full ISA ↔ Python line map for this kernel
```

rocgdb intercepts `hipModuleLoadData` via the HSA code-object notification
mechanism and reads the DWARF from the **in-memory HSACO blob** — no file on
disk is needed.

### The flydsl_gdb.py extension

`scripts/flydsl_gdb.py` adds four GDB commands that build on the DWARF line
table. `flydsl-step` is the most useful: it issues `stepi` in a loop until
`gdb.find_pc_line($pc)` returns a different Python source line, then prints
the context. This gives a "step-by-Python-line" experience over what is
actually ISA-level stepping.

### The flydsl_dwarf_mapper.py line mapper

`scripts/flydsl_dwarf_mapper.py` is a standalone offline tool. It extracts the
HSACO binary from the MLIR dump produced by `FLYDSL_DUMP_IR=1` (the
`gpu-module-to-binary` stage file contains the binary as an LLVM-escaped
string in the `bin= "..."` attribute) and parses the DWARF with `pyelftools`:

```bash
pip install pyelftools

# From the MLIR dump (FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/dump ...):
python3 scripts/flydsl_dwarf_mapper.py \
    /tmp/dump/<kernel>/20_gpu_module_to_binary.mlir \
    --save-hsaco kernel.hsaco

# Or from an already-extracted HSACO:
python3 scripts/flydsl_dwarf_mapper.py --hsaco kernel.hsaco
```

The output maps every Python source line to its ISA address range, which you
can then look up in `final_isa.s` (also in the dump dir) to find which
registers hold the values you care about before opening rocgdb.

### What you can and cannot do

| Capability | Available? |
|---|---|
| Break at kernel entry by symbol name | ✓ `break my_kernel_0` |
| Break at a specific Python source line | ✓ `break repro.py:52` |
| Step by Python source line | ✓ `flydsl-step` (wraps `stepi` in a loop) |
| `list` / see source context at current PC | ✓ `flydsl-where` |
| Read raw GPU registers (VGPRs/SGPRs/EXEC) | ✓ `flydsl-regs` |
| Per-wavefront / per-lane control | ✓ `thread apply all ...` |
| `print tid` — inspect by Python variable name | ✗ no variable DWARF |
| `print acc` — typed DSL value | ✗ types are not in LineTablesOnly |

To bridge the gap: use `flydsl_dwarf_mapper.py` to find which ISA address
range corresponds to the Python line you care about, read the `final_isa.s`
to see which registers are live there, then inspect those registers in rocgdb
with `flydsl-regs`. It is a two-step manual correlation, not a single `print`.

> **HIP/CK-Tile → FlyDSL.** `hipcc -g` gives source-level variable inspection
> (full DWARF with `DW_TAG_variable` entries). FlyDSL gives Python-line stepping
> with raw-register inspection (LineTablesOnly DWARF). Achieving named-variable
> DWARF would require the tracing pipeline to emit `llvm.dbg.value` intrinsics
> that associate SSA values with Python variable names — the infrastructure exists
> in MLIR but is not yet wired in.

### Practical ranking for wrong-results bugs

1. **`fx.printf`** guarded by `bid == 0 and tid == 0` — fastest to add and remove.
2. **Host-side bisection** — all-ones inputs, single tile, shrink grid.
3. **`flydsl_dwarf_mapper.py` + `final_isa.s`** — offline: find which ISA range
   corresponds to a suspect Python line, then read the registers manually.
4. **rocgdb + `flydsl_gdb.py`** — interactive: set a breakpoint on the suspect
   Python line, step through the GPU execution, inspect registers live.

## Can you debug the compiler *itself*?

Yes — FlyDSL is not a black box, and every layer is inspectable in text:

- **The assert names its own source.** Each C++ abort prints
  `path/to/File.h:line`. The repo builds from source (`scripts/build.sh`), so you
  can open that file, add an `llvm::errs()` or a tighter assert, and rebuild.
- **The IR is readable at every stage** (§2.8). `fly-opt` (in `tools/`) replays a
  single pass on a dumped `.mlir`, so you can minimize a failing case at the IR
  level rather than the Python level.
- **A debug build under a native debugger** works: `gdb --args python repro.py`
  (or `lldb`) breaks on the abort and gives a full C++ backtrace through the pass
  that failed.
- **Source-to-ISA mapping** — `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` threads your
  Python source locations into the emitted ISA (`.loc` directives in
  `final_isa.s`), so a faulting instruction maps back to your kernel line.

**Bottom line.** Trace-time and run-time bugs debug like ordinary Python and
ordinary GPU code. Compile-time aborts are the one rough patch — the message is a
C++ assertion, not a Python error — so you fall back to IR dumps and the
micro-repro loop of §8.5. But nothing is truly opaque: Python source, MLIR at
every stage, and the final ISA are all readable end to end.
