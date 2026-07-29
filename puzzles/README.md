# FlyDSL Puzzles

A hands-on, self-graded course for learning to write AMD GPU kernels in **FlyDSL**.
The puzzles start from the absolute basics (copy a tensor) and build up to an
**efficient GEMM pipeline**, then two real applications: **flash attention** and
**2D convolution**.

They are the practical companion to the [*FlyDSL for Dummies*](../docs/flydsl_for_dummies/)
book. Each puzzle references the book section that explains the concepts it exercises.

## Who this is for

Developers fluent in HIP C++ / CK-Tile C++ who want to learn the FlyDSL Python DSL.
You do **not** need prior Python-DSL experience, but you should be comfortable with
GPU concepts (wavefronts, LDS/shared memory, MFMA/tensor cores, tiling).

## Requirements

- An **AMD CDNA GPU**. The reference solutions target **gfx942 (MI300X) / gfx950 (MI350)**
  — CDNA MFMA, wave size 64.
- A working FlyDSL build (`import flydsl` succeeds). See the repo `CLAUDE.md` /
  `docs/installation.rst` for build instructions.
- PyTorch with ROCm (used only to generate reference outputs and check results).

## Layout

```
puzzles/
├── README.md                 # this file
├── common.py                 # shared torch-reference + allclose helpers
├── test_puzzles.py           # pytest harness (checks solutions vs torch)
├── puzzle01_copy.py          # ... puzzle skeletons (with TODO you fill in)
├── puzzle02_vector_add.py
├── ...
└── solutions/                # reference solutions (peek only after trying!)
    ├── puzzle01_copy.py
    └── ...
```

## How to work a puzzle

1. Open `puzzleNN_<name>.py`. Read the docstring: it states the problem, the
   concepts, hints, and the book section to read.
2. Fill in the code under the `# ==== YOUR CODE HERE ====` marker.
3. Run just that puzzle:

   ```bash
   python puzzles/puzzle01_copy.py
   ```

   Each skeleton has a `__main__` that runs your kernel and prints PASS/FAIL.
4. Stuck? Read the referenced book section, then peek at `solutions/`.

## Running the whole suite

The pytest harness imports the **reference solutions** and validates them against
torch. Use it to confirm your environment works, or (by editing the import at the
top of `test_puzzles.py`) to grade your own answers.

```bash
# Runs on the GPU; requires a CDNA device.
python -m pytest puzzles/test_puzzles.py -v

# A single puzzle:
python -m pytest puzzles/test_puzzles.py -k puzzle01 -v
```

## The progression

| # | Puzzle | Concepts | Book §|
|---|--------|----------|-------|
| **A. Warmup** ||||
| 01 | Copy a tensor | buffer tensors, tiled copy, partition, fragments | 5, 7 |
| 02 | Vector add (vectorized + predicated) | float4, OOB predication, register compute | 5, 7 |
| 03 | Scale & bias (elementwise + scalar) | scalar broadcast, elementwise math | 5 |
| 04 | 2D tiled copy | `zipped_divide`, thread/value layouts, `make_layout_tv` | 4, 5 |
| 05 | Transpose | layouts vs data movement, stride swap | 3, 4 |
| **B. Layout & reductions** ||||
| 06 | Row-sum reduction | wave `shuffle_xor` + LDS block reduce | 5 |
| 07 | Softmax | online max/sum, `exp2`, 3-pass reduction | 5 |
| 08 | RMSNorm | sum-of-squares reduce, `rsqrt`, gamma scale | 5 |
| **C. Efficient GEMM pipeline** ||||
| 09 | Single-tile MFMA GEMM | MMA atom, tiled MMA, fragments, `fx.gemm` | 6 |
| 10 | GEMM with a K-loop | `scf.for` with loop-carried accumulator | 6 |
| 11 | GEMM with LDS staging | LDS via `SharedAllocator`, g2s → s2r | 6 |
| 12 | GEMM double-buffered | ping-pong prefetch pipeline | 6 |
| 13 | GEMM swizzled + epilogue | bank-conflict swizzle, dtype-convert store | 6 |
| **D. Applications** ||||
| 14 | Flash attention (fwd) | online softmax over KV tiles, two GEMMs | 6 |
| 15 | 2D convolution | implicit GEMM, im2col-in-layout | 4, 6 |

> **Note on solutions.** The reference solutions are written to mirror the
> production kernels in `kernels/` and the `examples/` scripts. They are validated
> on gfx942 (MI300X) by `test_puzzles.py`.
