# WMMA-Driven Prefetch Pipeline 工作总结

> 文件: `kernels/gemm_fp8fp4_gfx1250.py`, `kernels/pipeline_utils.py`
> 测试: `tests/kernels/test_gemm_fp8fp4_gfx1250.py`
> 复现脚本: `dump_quad_isa.sh`

---

## 1. 原始问题（已修复并验证 ✅）

### 现象
`dump_quad_isa.sh` 跑 `PF_QUADRANT=1 PF_PIPELINE=1` 精度不对，但 cmodel 仿真精度正常。

### 根因
K-split pipeline 每个 tile 发 **2 个 TDM**（pf 段 + remain 段）。`s_wait_tensorcnt` 是 **FIFO 计数**：`s_wait_tensorcnt(N)` 保证最老的 `(已发总数 - N)` 个 TDM 完成。

旧代码 `_pf_keep = 2*num_buffers - 1`（N=3 时 = 5）太大：
- 正在被消费的 buffer 的 pf+rem TDM 是在 `2*(num_buffers-1)` 个 TDM 之前发出的
- `s_wait_tensorcnt(5)` 时这些 TDM 一个都不保证完成 → WMMA 读到还没 load 完的 LDS → **数据竞争**

cmodel 把 TDM 当**同步**处理（瞬间完成），所以任何 `s_wait_tensorcnt` 值都过 —— 代码注释已写明 "cmodel treats TDM as synchronous, NOT cmodel-validatable"。这就是为什么仿真过、真机挂。

### 修复（已验证）
```python
_pf_keep = max(0, 2 * (num_buffers - 1) - 2)   # N=3 → 2, N=4 → 4, N=2 → 0
```

**验证结果**（K=3072, num_buffers=3, pf-depth-wmma=4）：
- `_pf_keep=2` → PASS
- `_pf_keep=3` → FAIL
- 确认 `2*(N-1)-2` 是正确的最大安全值
- legacy 路径（`PF_FULL_PREFETCH` 不设）现在稳定通过

### 验证手段
- `FORCE_TCNT0=1`（在 `python/flydsl/expr/rocdl/tdm_ops.py:1123` 的 `tensor_wait`）强制所有 `s_wait_tensorcnt` 为 0 → 必过，证明是 tensorcnt 问题
- ISA dump 确认 `s_wait_tensorcnt 0x5` 出现在 steady fence

---

## 2. 全预取模式 PF_FULL_PREFETCH（进行中，未通过 ⚠️）

### 动机
`num_buffers=2` 时旧架构 `pre_loaded = N-1 = 1` 只预取 1 个 buffer。K-split 下 `_pf_keep = max(0, 2*(2-1)-2) = 0`，每个 fence 都 `s_wait_tensorcnt(0)` 完全堵死，没有 TDM overlap。

### 设计（用户给定）
- `pre_loaded = num_buffers`：prologue 装满全部 N 个 buffer
- 主循环 **先消费再 backfill**：`load_stage = buf_idx`（回填刚消费完的 buffer）
- 利用 K-split 的 pf/remain 分段做 partial overlap（pf 段和 remain 段是 LDS 中不相交的 K 区域）
- `_pf_keep = 2*(num_buffers-1)`：每个 phase 边界发 1 个 TDM 后 drain 到 2*(N-1)，刚好淘汰 N-1 tiles 前的同类型 TDM（下一 phase 要读的那块）

用户给定的 TDM 流程：
```
Prologue:
  Prefetch first tile (完整 TDM); tensorcnt(0)
  Prefetch N-1 more (完整 TDM); ds_load(pf_wmma_deep); tensorcnt(0)
主循环:
  Phase 1 (消费 pf carry, interleave remain ds_loads)
    Issue next_pf TDM; tensorcnt(2*(N-1)); s_barrier
  Phase 2 (消费 remain, interleave next_tile pf ds_loads)
    Issue next_remain TDM; tensorcnt(2*(N-1)); s_barrier
Tail (last N buffers, 倒序):
  Phase 1: tensorcnt(2*(N-1) - 1); s_barrier
  Phase 2: tensorcnt(2*(N-1) - 2); s_barrier
  ... 每个 fence 递减 1 直到 0
```

### 三 phase 的情况（用户重点提示，疑似 bug 来源）
WMMA i 发 ds_load for position `j = i + D`：
- `j < A`：读**当前 tile** 的 remain LDS → cur_all
- `j >= A`：读**下一 tile** 的 pf LDS → nxt_raw

当前测试参数 `pf-depth-wmma=4`：`wmma_m_rep=2, wmma_n_rep=4, k_wmma_steps=2`
→ `_pf_wpks=8, _A_wmma=16, _pf_D=4, _pf_Dk=1`（pf seg=ks0, remain seg=ks1）

因为 `D=4 < A-D=12`，这是 **3-phase** 情况：
- Phase 1 (i=0..3): 消费 carry, load cur_remain[4..7]
- i=4: **cb_pf** 触发，issue pf TDM 写 buf[X] ks0
- Phase 2 (i=4..11): 消费 cur_remain, load cur_remain[8..15]（j=8..15 仍 < A=16）
- Phase 3 (i=12..15): 消费 cur_remain, load next_pf[0..3]（j=16..19 >= A）
- i=16(末): **cb_rem** 触发，issue remain TDM 写 buf[X] ks1

### 已做的代码修改

**`kernels/pipeline_utils.py`** — `make_tail_plan` 加 `full_prefetch=False` 参数：
```python
def make_tail_plan(num_buffers, pre_loaded, extra, full_prefetch=False):
    ...
    if i < extra:
        load_stage = compute_stage if full_prefetch else (i + num_buffers - 1) % num_buffers
    else:
        load_stage = None
```

**`kernels/gemm_fp8fp4_gfx1250.py`**:
- 行 305-310:
  ```python
  _full_pf_req = os.environ.get("PF_FULL_PREFETCH", "0") == "1"
  pre_loaded = num_buffers if _full_pf_req else (num_buffers - 1)
  ...
  _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra, full_prefetch=_full_pf_req)
  ```
- 行 ~3221: `_full_prefetch = _full_pf_req and _pf_pipeline`
- 行 ~3228: `_pf_keep`（注意：**必须用 `if const_expr(_full_prefetch):` 不能用 Python ternary**，否则 AST rewriter 报 `_pf_keep not defined`）
  ```python
  if const_expr(_full_prefetch):
      _pf_keep = 2 * (num_buffers - 1)
  else:
      _pf_keep = max(0, 2 * (num_buffers - 1) - 2)
  _pf_keep_cb_rem = _pf_keep
  ```
- 行 ~3442: 主循环 `load_stage`（同样用 `const_expr`）
  ```python
  for buf_idx in range_constexpr(num_buffers):
      if const_expr(_full_prefetch):
          load_stage = buf_idx
      else:
          load_stage = (buf_idx + num_buffers - 1) % num_buffers
  ```
- 行 ~3475: `_tdm_cb_rem` 在 full-PF 下**每个** buf_idx 都 fence（legacy 只在最后一个）：
  ```python
  def _tdm_cb_rem(_nb=load_stage, _bi=buf_idx):
      _issue_active_tdm_seg(_nb, _addr_rem_box[0], 1)
      _addr_rem_box[0] = _addr_rem_box[0] + active_adv_i32
      if const_expr(_full_prefetch):
          _pipeline_fence(outstanding=_pf_keep_cb_rem)
      elif const_expr(_bi == num_buffers - 1):
          _pipeline_fence(outstanding=_pf_keep_cb_rem)
  ```
- 行 ~3713: tail drain 从 `_pf_keep_cb_rem` 开始递减
  ```python
  _tail_tcnt_box = [_pf_keep_cb_rem]
  ```
- 行 3394: prologue 的 split signal 在 full-PF 下跳过
  ```python
  if const_expr(loop_iters > 0 and use_ws_tdm_split_signal_overlap and not _full_prefetch):
  ```

**注意**：steady fence 的去除曾试过又加回（行 ~3486）。当前状态是 **保留 steady fence**（signal+wait），因为去除后并不能修复 bug。

### 当前状态：未通过 ❌
| 配置 | 结果 |
|---|---|
| K=768（3 tiles = 纯 prologue+tail, `loop_iters=0`） | **PASS** |
| K=1536（6 tiles, `loop_iters=1`） | **FAIL** |
| K=1536 + `FORCE_TCNT0=1` | **FAIL** |
| K=1536 + `FORCE_TCNT0=1` + `PF_FORCE_WAIT=1` | **FAIL** |

**关键判断**：`FORCE_TCNT0=1`（所有 tensorcnt 强制 0）**仍然失败**，已用 ISA dump 确认 15 个 `s_wait_tensorcnt` 全是 `0x0`。
→ **不是 timing/tensorcnt 问题，是数据流/结构性 bug。**

`loop_iters=0`（纯 tail）过、`loop_iters=1`（有主循环）挂 → **bug 在主循环**。

### 已排除的可能性
- ✅ Tile 总数正确（legacy 和 full-PF 都是 6 tiles，计算顺序 0→5 一致）
- ✅ Tail plan 正确（`make_tail_plan(3,3,0,full_prefetch=True)` = `[(None,0,0),(None,1,0),(None,2,-1)]`）
- ✅ Carry 链推导一致（手工追踪两种模式的 nxt_raw carry，结论相同）
- ✅ pf/remain 是 LDS 不相交区域（`_stage_lds_sgprs` vs `_stage_lds_rem_sgprs`），cb_pf 写 ks0 不冲突 remain WMMA 读 ks1
- ✅ Barrier 不是缺失（加回 steady fence 仍挂）
- ✅ ISA 行数/TDM 数与 legacy 完全相同（915 行, 12 个 tensor_load_to_lds）

### 下一步排查方向（机器恢复后）
1. **用户重点怀疑 3-phase**：测 `pf-depth-wmma=8`（D=8=A/2 → 2-phase）和 `pf-depth-wmma=12`（D>A-D → 另一种 3-phase）对比 `pf-depth-wmma=4`（D<A-D → 3-phase），定位是否 3-phase 特有
   ```
   FLYDSL_RUNTIME_ENABLE_CACHE=0 PF_QUADRANT=1 PF_PIPELINE=1 PF_FULL_PREFETCH=1 FORCE_TCNT0=1 \
     python3 tests/kernels/test_gemm_fp8fp4_gfx1250.py --data-format a8w4 --scale-mode mxscale \
     -M 1 -N 12288 -K 1536 --tile-m 32 --tile-n 256 --tile-k 256 \
     --m-warp 1 --n-warp 4 --num-buffers 3 --split-k 1 --pf-depth-wmma 8 \
     --cluster-m 1 --cluster-n 1 --l2-prefetch-distance 0 --out-dtype bf16 --fill-mode random
   ```
2. **diff 主循环 ISA**：`/tmp/isa_legacy` vs `/tmp/isa_fullpf`（FORCE_TCNT0 归一化 tensorcnt 后），重点看 `tensor_load_to_lds` 的 LDS base 寄存器和 DRAM 地址寄存器、ds_load 的 offset，确认 full-PF 主循环每个 cb 写的 buffer/地址是否真的对
3. **怀疑点**：主循环 cb 在 full-PF 下写 `load_stage=buf_idx`（正在消费的 buffer）。虽然 pf/remain 区域理论不相交，但要确认 `_issue_active_tdm_seg(buf_idx, _addr_pf_box[0], 0)` 的 LDS base 和 compute 读的 carry/cur_all LDS base 是否真的是同一 buffer 的不同 K 区域，而非整个 buffer 重叠
4. **怀疑点**：`_pl_addr_pf_box`/`_pl_addr_rem_box` 初始化用 `active_addr_lo`/`_seg_alo_rem_pl`，这俩在 prologue 循环里被推进了 `pre_loaded` 次。确认 full-PF 下 box 起始地址 = tile N（而非 tile N-1）是否与主循环消费的 buffer 数据匹配

### 验证命令（机器恢复后回归）
```bash
# 原始问题修复回归（必须过）
FLYDSL_RUNTIME_ENABLE_CACHE=0 PF_QUADRANT=1 PF_PIPELINE=1 \
  python3 tests/kernels/test_gemm_fp8fp4_gfx1250.py --data-format a8w4 --scale-mode mxscale \
  -M 1 -N 12288 -K 3072 --tile-m 32 --tile-n 256 --tile-k 256 \
  --m-warp 1 --n-warp 4 --num-buffers 3 --split-k 1 --pf-depth-wmma 4 \
  --cluster-m 1 --cluster-n 1 --l2-prefetch-distance 0 --out-dtype bf16 --fill-mode random

# 全预取（目标，当前挂）
... 同上加 PF_FULL_PREFETCH=1
```
```

---

## 环境变量速查
| 变量 | 作用 | 位置 |
|---|---|---|
| `PF_QUADRANT=1` | 路由 quadrant shape 到 pf pipeline | gemm:557 |
| `PF_PIPELINE=1` | 启用 K-split WMMA-driven pipeline | gemm:3219 |
| `PF_FULL_PREFETCH=1` | (新增) 全预取模式 pre_loaded=N | gemm:305 |
| `FORCE_TCNT0=1` | 强制所有 s_wait_tensorcnt=0（诊断） | tdm_ops.py:1123 |
| `PF_FORCE_WAIT=1` | 每个 WMMA 前强制 s_wait_dscnt(0)（诊断） | gemm:1810 |
| `pf_depth_wmma` (CLI `--pf-depth-wmma`) | prefetch 深度 D（WMMA 数） | gemm:705 |
