# CuTe IR Lowering Pass Pipeline


```
Python CuTe DSL
      ↓ (Python codegen)
┌─────────────────────────────────────────┐
│  cute_ir (Hardware-Agnostic)            │
│  - Layout algebra operations            │
│  - Type: Layout, Tile, Tensor           │
└─────────────────────────────────────────┘
      ↓ cute-layout-canonicalize
      ↓ cute-layout-fusion
┌─────────────────────────────────────────┐
│  Optimized cute_ir                      │
└─────────────────────────────────────────┘
      ↓ cute-tensor-partition
      ↓ cute-memory-coalescing
┌─────────────────────────────────────────┐
│  cute_nvgpu_ir (GPU-Aware)              │
│  - MmaAtom, CopyAtom, TMA               │
│  - Warpgroup/Warp operations            │
└─────────────────────────────────────────┘
      ↓ cute-nvgpu-to-nvgpu
      ↓ cute-async-pipeline
┌─────────────────────────────────────────┐
│  MLIR nvgpu dialect                     │
│  - nvgpu.warpgroup.mma.sync             │
│  - nvgpu.tma.async.load                 │
└─────────────────────────────────────────┘
      ↓ nvgpu-to-nvvm
      ↓ nvvm-to-llvm
      ↓ gpu-to-llvm
      PTX → SASS
```

## Pass 调用示例

```bash
mlir-opt input.mlir \
  --cute-layout-canonicalize \
  --cute-nvgpu-to-nvgpu="target-arch=sm_90" \
  --cute-async-pipeline="pipeline-depth=2" \
  --convert-nvgpu-to-nvvm \
  -o output.mlir
```
