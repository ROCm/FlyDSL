# MoE Stage-1 Megakernel 两项优化：B-load cache modifier (b_nt) 与 dispatch payload 去重 (dedup)

本文档记录对 stage-1 融合 megakernel（`kernels/fused_moe_megakernel.py` 的 `FusedMoEMegaStage1`，
其 GEMM 由 `kernels/fused_moe_gemm_2stage.py::compile_fused_moe_gemm1` 编译）新增的两个优化点的
思路、做法、关键代码位置，以及在 8×MI355X 上通过 CUDAGraph 测得的 bs=32/64/128/256/512 前后性能对比。

两项优化均为 **数值无损**（下文所有 batch 的 `max|Δ|` 相对 baseline 输出均为 `0.00e+00`，逐位一致）。

---

## 优化点 1：B-load cache modifier（`b_nt`）

### 思路

Stage-1 GEMM 每个 tile 要从显存读权重 B（gate/up，fp4）。硬件的 `buffer_load` 支持 **cache modifier**：

- `b_nt=0`：正常缓存（把 B 读进 L1/L2，供后续复用）；
- `b_nt=3`：**streaming / non-temporal**（绕过/优先淘汰，不长期占用 cache 行）。

关键观察:一个 expert 的权重是否会在它的多个 M-tile 间被**复用**,取决于该 expert 分到的 token 数(≈ per-rank
token 数 `mtpr`):

- **decode / 中小 batch**（`mtpr ≤ 256`）:每个 expert 的 M-tile 少,权重基本读一次就不再用 → 用 `b_nt=0` 反而
把"用不到的权重"塞进 L2 污染缓存;改用 **streaming(`b_nt=3`)** 更优。
- **prefill / 大 batch**（`mtpr ≥ 512`）:权重会被同一 expert 的多个 M-tile 复用 → 保持 **缓存(`b_nt=0`)** 更优。

因此采用启发式默认值 `b_nt = 3 if mtpr <= 256 else 0`(实测拐点在 256 与 512 之间,见下),并可用 env
`FUSED_MEGA_B_NT` 或构造参数覆盖。

### 做法与关键代码位置

- **kernel 侧(builder,已存在支持,补齐了 cache 命名)**:`compile_fused_moe_gemm1(b_nt=...)` 把 `b_nt` 作为
B-load 的 `cache_modifier`。
  - 参数 `b_nt`:`kernels/fused_moe_gemm_2stage.py:211`。
  - B-load 使用 `cache_modifier=b_nt`:`kernels/fused_moe_gemm_2stage.py`(GEMM 主循环 B-load）。
  - **关键修复**:kernel 变体名原先**不含 b_nt**,不同 `b_nt` 会在编译缓存中撞名导致优化静默失效。已在 kernel 名
  追加 `_bnt{b_nt}`:`kernels/fused_moe_gemm_2stage.py:502`。
- **host 侧(`FusedMoEMegaStage1`)**:
  - 启发式 `_default_b_nt(mtpr)`:`kernels/fused_moe_megakernel.py:55`。
  - 优先级解析（显式参数 > env `FUSED_MEGA_B_NT` > 启发式）:`kernels/fused_moe_megakernel.py:283-287`。
  - 传给 compile:`kernels/fused_moe_megakernel.py:353`（`b_nt=self._b_nt`)。
  - 新增构造参数 `b_nt=None`。

---

## 优化点 2：dispatch payload 去重（`dedup`）

### 思路

MoE dispatch 阶段,本 rank 的每个 token 会按 topk 路由到多个 expert。当一个 token 的**多个 topk expert 落在同一个
目标 GPU**上时,原实现会把这个 token 的 **7KB 激活重复发送/写入多份**(expert-major:每个 expert 拷贝一份)。

去重的做法:producer 端对每个源 token 到某个目标 GPU **只写一次**激活(写进一个 **token-major** 的接收 buffer
`rx_tok`,行号 = 全局源 token id `src_global`);GEMM 端 A-gather 时按 srcmap 解出 `src_global`,从 `rx_tok[src_global]`
读回。这样把"同一目标卡上的重复激活写入"从 `(该 token 落在该卡的 expert 数)` 份降到 **1 份**,省下 dispatch 的
payload 写带宽。元数据(idx/wts/srcmap/scale,均很小)仍按 expert-major 原样写,GEMM tile 仍能通过 srcmap 找到
`src_global`,故**数学完全等价**。

### 做法与关键代码位置

- **kernel 侧(builder,新增 `dedup` 路径)**:`kernels/fused_moe_gemm_2stage.py`
  - 新增参数 `dedup: bool`:`:227`;`_dedup = dedup and _atom_contract`(去重依赖 srcmap/atom 契约):`:441`。
  - kernel 变体名追加 `_dd`:`:502`。
  - fixedslot dispatch prologue 读 token-major buffer 的 P2P 表(host 追加在 disp 槽 26):
  `_p_rx_tok = _dp(26)`,`:789`。
  - **producer 写一次**(§13.4.13 first-dest dedup):算 `_is_first_dest`,gate 拷贝,`_dedup` 时目标为
  `rx_tok[dest_pe][src_global]`,否则原 expert-major `rx[slot]`:`kernels/fused_moe_gemm_2stage.py:882-925`
  附近(注释 `first-dest dedup: write the 7KB activation ONCE`,`:884`)。
  - **GEMM A-gather**:`_dedup` 时 A row = `rx_tok[src_global]`,`src_global` 由 disp[19] 的 LOCAL srcmap 在该 tile 行
  解出(低 24 位),保留 sorted 参数用于 atom 输出:`kernels/fused_moe_gemm_2stage.py:2030` 附近。
- **host 侧(`FusedMoEMegaStage1`)**:去重接线(env `FUSED_MEGA_DEDUP=1` 或构造参数 `dedup=True`):
`kernels/fused_moe_megakernel.py:419-457`
  1. 分配 token-major 对称 buffer `rx_tok` + P2P 表 `p2p_rx_tok`（`:429`）;
  2. 用 `dedup=True` 重编译 GEMM（`:449`);
  3. 把 `p2p_rx_tok` 追加到 disp 表槽 26（`:453`);
  4. 把 A-input 视图 `_rx` 重指向 `rx_tok`（`:456`）。
  - 新增构造参数 `dedup=None`(默认读 env `FUSED_MEGA_DEDUP`)。

> 说明:去重仅在 `atom_contract`(srcmap 契约)且非 compact 布局下启用;本 bench 的 bs≤512 均为 fixedslot 布局。

---

## 测试方法（CUDAGraph）

- 脚本:`tests/kernels/bench_mega_stage1_opt.py`（自包含;复用 `bench_moe_intranode_stage1_groupgemm.py`
的 `_prepare`/`_setup_dist`）。
- 被测算子:`kernels/fused_moe_megakernel.py::FusedMoEMegaStage1`（stage-1 单发核）。
- 对每个 batch 构造 4 个变体 op:`baseline(b_nt=0,dedup=off)` / `+b_nt(b_nt=3)` / `+dedup` / `+both`,
每个 op 用 `torch.cuda.CUDAGraph` 捕获 `forward` 后 replay 计时(warmup 8、iters 30,跨 rank barrier 同步)。
- 正确性:对比各变体的 `_out` 与 baseline 的 `max|Δ|`。
- 运行(8×MI355X,v4_pro a8w4:model_dim=7168, inter_dim=3072, experts=384, topk=6, ep=8):

```bash
cd FlyDSL
export PYTHONPATH=$PWD:$PWD/tests:mori/python:aiter
export AITER_USE_SYSTEM_TRITON=1 MORI_SHMEM_HEAP_SIZE=40G
for BS in 32 64 128 256 512; do
  torchrun --standalone --nproc_per_node=8 --master-port $((29900+BS)) \
    tests/kernels/bench_mega_stage1_opt.py --network v4_pro --quant a8w4 \
    --bs-list $BS --iters 30 --warmup 8
done
```

---

## 性能结果（v4_pro a8w4,per-rank bs,CUDAGraph device time）

CUDAGraph 单次 forward device 时间(ms),以及相对 baseline 的加速比;`max|Δ|` 为相对 baseline 输出的最大绝对误差。


| bs (per-rank) | baseline (ms) | +b_nt               | +dedup              | +both               | max|Δ| |
| ------------- | ------------- | ------------------- | ------------------- | ------------------- | ------ |
| 32            | 0.2389        | 0.2213 (**1.079x**) | 0.2333 (1.024x)     | 0.2158 (**1.107x**) | 0      |
| 64            | 0.2436        | 0.2270 (**1.073x**) | 0.2413 (1.009x)     | 0.2198 (**1.108x**) | 0      |
| 128           | 0.2538        | 0.2346 (**1.082x**) | 0.2495 (1.017x)     | 0.2299 (**1.104x**) | 0      |
| 256           | 0.2887        | 0.2742 (1.053x)     | 0.2805 (1.029x)     | 0.2624 (**1.100x**) | 0      |
| 512           | 0.3921        | 0.4016 (0.976x)     | 0.3809 (**1.029x**) | 0.3935 (0.997x)     | 0      |


### 分优化点收益

**b_nt（streaming B-load）**

- bs=32/64/128:**+7.3% ~ +8.2%**(小 batch,权重不复用,streaming 避免污染 L2)。
- bs=256:**+5.3%**(streaming 仍明显赢)。
- bs=512:**−2.4%**(大 batch,权重被同 expert 多 M-tile 复用,streaming 反而把它们挤出 cache)。
- ⇒ 实测**拐点在 256 与 512 之间**,故启发式取 `b_nt = 3 if mtpr<=256 else 0`:mtpr≤256 开 streaming(含 256 的
+5.3%),mtpr≥512 保持 cache(避开 512 的 −2.4% 回退)。按启发式部署时,bs≤256 自动 b_nt=3,bs≥512 自动 b_nt=0。

**dispatch dedup**

- 各 batch **稳定 +1.0% ~ +2.9%**(bs=32:+2.4%,bs=256/512:+2.9%),随 batch 略增。收益来自省掉"同目标卡上
同一 token 的重复 7KB 激活写入";收益幅度取决于 topk 落在同卡的碰撞率(v4_pro topk=6 / ep=8)。

**both（按 batch 组合）**

- bs≤128:两者叠加 **≈ +10.4% ~ +10.8%**。
- bs=256:**+10.0%**。
- bs=512:≈ 持平(b_nt 的回退与 dedup 的收益相抵);此档实际部署应只开 dedup(启发式令 b_nt=0),即 **+2.9%**。

### 启发式部署配置(`b_nt = _default_b_nt(mtpr)`,即 mtpr≤256→3,否则 0)

上表 `+b_nt/+both` 为**强制 b_nt=3** 以隔离 b_nt 在各 batch 的效果(故 bs=512 出现 −2.4% 回退)。实际部署走
启发式:bs≤256 自动 b_nt=3,bs≥512 自动 b_nt=0。对回退档 bs=512 用启发式重测(`+*(heur)` 表示 b_nt=启发式值):


| bs=512 变体       | ms     | speedup                         | max|Δ| |
| --------------- | ------ | ------------------------------- | ------ |
| baseline        | 0.3874 | 1.000x                          | 0      |
| +b_nt(heur=0)   | 0.3905 | 0.992x(≈baseline,同 kernel,测量噪声) | 0      |
| +dedup          | 0.3764 | 1.029x                          | 0      |
| **+both(heur)** | 0.3754 | **1.032x**                      | 0      |


⇒ 启发式部署下 **bs=512 无回退**(+3.2%,全部来自 dedup;b_nt=0 与 baseline 同核)。即"启发式 b_nt + dedup"在
全 batch 都 ≥ baseline:小 batch ~+10%,大 batch ~+3%。

### 结论

- **b_nt** 是 decode/中小 batch 的主要收益来源(~~+5~~8%,含 bs=256 的 +5.3%),但必须按 `mtpr` 启发式开关
(`b_nt=3 if mtpr<=256 else 0`),否则会在大 batch(如 bs=512 −2.4%)造成回退;且必须带 kernel 名 tag(否则
编译缓存撞名、优化失效)——本次已修复。
- **dedup** 是全 batch 稳定的小幅收益(~~+1~~3%),数值无损。
- 二者正交、可叠加;**启发式部署**下:bs≤256 组合 **~1.10x**,大 batch(512)**+3.2% 无回退**。全部 bit-exact。

