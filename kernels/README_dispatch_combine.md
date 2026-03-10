# FlyDSL Dispatch/Combine IntraNode — 使用说明

## 文件结构

```
kernels/
├── dispatch_combine_intranode_v2.py      # v2 Kernel 工厂（Python FlyDSL 语法）★ 主文件
├── dispatch_combine_intranode_op_v2.py   # v2 算子包装器（对外 API）              ★ 主文件
├── dispatch_combine_intranode_kernel.py  # v1 Kernel（手写 LLVM IR，Legacy）
└── dispatch_combine_intranode_op.py      # v1 算子包装器（Legacy）

tests/kernels/
└── test_v2_accuracy_perf.py             # v2 精度 + 性能测试脚本                  ★ 测试入口
```

---

## 精度测试

### 快速启动（spawn 模式，适合 2–8 GPU）

```bash
cd /home/yashao/FlyDSL
export MORI_SHMEM_HEAP_SIZE=6G
export LD_LIBRARY_PATH=/home/yashao/FlyDSL/python/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

# EP=2，精度测试（与 mori 参考对比）
python tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --world-size 2 \
    --max-tokens 64 \
    --hidden-dim 128 \
    --k 2 \
    --num-experts 8

# EP=8，精度测试
python tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --world-size 8 \
    --max-tokens 64 \
    --hidden-dim 128 \
    --k 4 \
    --num-experts 64
```

### torchrun 模式（适合生产环境）

```bash
torchrun --nproc_per_node=8 tests/kernels/test_v2_accuracy_perf.py \
    --mode test \
    --max-tokens 256 \
    --hidden-dim 512 \
    --k 4
```

### 精度输出说明

```
[精度测试] EP=2, max_tok=64, hidden=128, k=2, experts=8
  Round 1: cur_tok=39  total_recv v2=67 ref=67 ✓
    dispatch:  tok max_diff (v2 vs ref, sorted)      = 0.0000 ✓
    combine:   v2/k vs ref (should match)           = 0.xxxx ✓
    combine:   v2/k vs original_input                = ~x.xxxx (expected: (1-n_valid/k)*max_orig)
```

- **dispatch max_diff = 0.0000** ✓ 表示 dispatch 完全正确（与 mori 参考比特级一致）
- **combine v2/k vs orig** 的差值是预期行为：v2 combine 做无权重累加，`out = n_valid * inp`；
  对于 npes=2, k=2 场景，期望差值约为 `0.5 * max_inp`（每 token 平均 1 个非重复专家）

---

## 性能测试

### 单次运行（精度 + 性能）

```bash
python tests/kernels/test_v2_accuracy_perf.py \
    --mode both \
    --world-size 2 \
    --max-tokens 256 \
    --hidden-dim 512 \
    --k 4 \
    --num-experts 16 \
    --block-num 8 \
    --warp-per-block 4 \
    --warmup 3 \
    --iters 10
```

### 只跑性能

```bash
python tests/kernels/test_v2_accuracy_perf.py \
    --mode bench \
    --world-size 8 \
    --max-tokens 4096 \
    --hidden-dim 7168 \
    --k 8 \
    --num-experts 256 \
    --block-num 80 \
    --warp-per-block 16
```

### 性能输出说明

```
[性能测试] EP=2, max_tok=256, hidden=512, k=4, block_num=8, wpb=4
                    FlyDSL v2     mori ref          对比
  ────────────── ──────────── ────────────  ──────────
  dispatch           2.132 ms     0.135 ms  (0.06x ref)
  combine            0.230 ms     0.113 ms  (0.55x ref)
  dispatch+comb      2.362 ms     0.248 ms  (0.11x ref)
```

---

## 全参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `both` | `test`/`bench`/`both` |
| `--world-size` | 2 | GPU 数量（EP 度） |
| `--max-tokens` | 512 | 每 rank 最大 token 数 |
| `--hidden-dim` | 512 | Hidden dimension |
| `--k` | 4 | Top-k 专家数 |
| `--num-experts` | ws×32 | 专家总数（若不指定则自动推导） |
| `--block-num` | 16 | CUDA block 数 |
| `--warp-per-block` | 4 | 每 block 的 warp 数 |
| `--chip` | `gfx942` | AMD GPU 型号 |
| `--n-rounds` | 3 | 精度测试轮数 |
| `--warmup` | 3 | 性能测试预热轮数 |
| `--iters` | 10 | 性能测试正式轮数 |
| `--no-compare` | — | 跳过与 mori ref 的对比 |

---

## 在 Python 代码中直接使用

```python
import torch
from kernels.dispatch_combine_intranode_op_v2 import (
    FlyDSLDispatchCombineConfigV2,
    FlyDSLDispatchCombineIntraNodeOpV2,
)

cfg = FlyDSLDispatchCombineConfigV2(
    rank=rank,
    world_size=world_size,
    hidden_dim=7168,
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,   # 256 experts / 8 GPUs
    num_experts_per_token=8,
    data_type=torch.bfloat16,
    warp_num_per_block=16,
    block_num=80,
    chip="gfx942",
)

op = FlyDSLDispatchCombineIntraNodeOpV2(cfg)

# 每轮推理
op.reset()
dispatch_out, wts, _, idx, total_recv = op.dispatch(inp, weights, None, expert_indices)

# expert 处理 ...

combine_out, _ = op.combine(expert_output, weights, idx)
```

---

## 常见问题

**Q: 编译时间很长？**  
A: 首次编译约 15-30 秒，HSACO 会缓存在 `/tmp/v2_disp_*.hsaco`，后续重启进程可直接加载。
缓存文件名包含配置参数，不同配置不会互相污染。

**Q: `hipModuleLaunchKernel` 返回 error 719（资源不足）？**  
A: 减小 `--warp-per-block`（如从 8 改为 4）或 `--block-num`，降低每 block 的线程数。

**Q: dispatch max_diff 不是 0？**  
A: 检查精度测试是否跨轮次（多轮精度测试时存在 shmem barrier 配对问题），建议单轮（`--n-rounds 1`）验证。
