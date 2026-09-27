# Shared/reuse MLA + MoE kernel

[English](PERFORMANCE.md)

本目录包含一个固定 GLM-5 MLA + MoE 分片的 FlyDSL 实现。生产路径通过
`SharedReuseMlaMoeLayer` 提供，并由 `build_shared_reuse_kernel` 生成。该路径不导入
TileRT kernel 主体，也不嵌入汇编。原有的汇编捕获、改写和启动桥接代码已全部删除。

`native_baseline.py` 仍可选择性依赖 TileRT，用于把同一组生成权重转换给已发布的
TileRT wrapper，从而直接比较两个实现。FlyDSL 执行路径不会导入该适配器。

## 固定分片与计算模式

每个 rank 固定使用 8 个 attention heads、专家 intermediate size 256、hidden size
6144、256 个 routed experts、1 个 shared expert 和 top-8 routing。因此二卡、四卡
结果测量的是相同固定分片配合较小通信组，不是完整模型 TP2 或 TP4 的形状。

公开的 `MoeMode` 包含：

| 模式 | 专家 activation | 专家 weight | Up/gate 到 down 的交接格式 |
|---|---|---|---|
| `w8a8` | 每 128 个元素动态量化 FP8 E4M3 | block-scaled FP8 E4M3 | FP8 |
| `w8a16` | BF16 | block-scaled FP8 E4M3 | BF16 |
| `a16w4` | BF16 | 每 1x32 使用 E8M0 scale 的 MXFP4 | BF16 |
| `a8w4` | 每 1x32 使用 E8M0 scale 的 MXFP8 E4M3 | 每 1x32 使用 E8M0 scale 的 MXFP4 | MXFP8 |

所有模式的 attention weights 都保持 block-scaled FP8。当前支持 sample count 1、2、4、8
以及 peer count 1、2、4、8。host wrapper 会在分配 GPU buffer 前验证完整固定分片约束。

## ATOM 存储布局审计

MXFP4 模式现在默认采用可以直接消费 ATOM tensor、无需重新打包 expert tensor 的布局：

| Tensor | 逻辑顺序 | MXFP4 默认路径的物理存储 | 与 ATOM 的兼容性 |
|---|---|---|---|
| `w_ug` | 256 个 routed experts，随后是 shared expert 256；gate rows 在前、up rows 在后 | AITER `shuffle_weight(..., layout=(16, 16), is_guinterleave=False)` | 完全一致 |
| `w_dn` | 相同 expert 编号；output rows x input-intermediate columns | 相同 AITER 16x16 shuffle | 完全一致 |
| `s_ug`、`s_dn` | 每个 1x32 block 一个 E8M0 byte | AITER non-interleaved `shuffle_scale` | 完全一致 |
| `w_r` | 256 行 BF16 router | 默认仍用 mono-kernel MFMA packing；可选 ATOM row-major | 逻辑一致，默认不能零拷贝 |
| MLA projection weights | `qkv_a=[q_a; kv_a]`；每个 `q_b` head 为 `[nope; rope]` | 现有 block-128 FP8 MFMA packing | 与 `amd/GLM-5.1-MXFP4` 的 BF16 attention weights 不兼容 |
| KV cache | 每个 token 为 `[k_c(512), k_pe(64)]` | 单个连续 BF16 `[tokens, 576]` tensor | 对齐 benchmark 中 shape、stride、dtype 完全一致；ATOM serving 可通过配置选择 FP8 |

expert value 和 scale packer 已针对 GLM-5 up/gate、down 两种形状，与当前 AITER 的
`shuffle_weight`、`shuffle_scale` 做过逐字节比较，结果完全一致。norm vectors 和
routing bias 原本就是普通连续 tensor。ATOM 默认的 `ATOM_MOE_GU_ITLV=0` 也与本
kernel 的 gate-then-up 行顺序一致。

对齐后的 ATOM layer 逻辑 expert tensor shape 为
`w13=[257,512,3072]`、`w13_scale=[257,512,192]`、`w2=[257,6144,128]`、
`w2_scale=[257,6144,8]`。经过 AITER 后处理后，value shape 不变，scale buffer
变为 `[131584,192]` 和 `[1579008,8]`。这些 shape 以及每个 buffer 的字节内容都与
FlyDSL 默认 MXFP4 路径一致。

保留 `--router-weight-layout atom` 用于零拷贝实验，但不设为默认：在其他存储均采用
ATOM-compatible 的 A16W4 下，TP1 S=1 从 31.19 降到 31.08 us/layer，略有提升；但
S=8 从 70.80 上升到 72.47 us/layer，回退 2.4%。ATOM 的独立 GEMM 可以在内部对
row-major router 做 tiling，而 mono-kernel 的直接 MFMA load 会跨越 6144-element
row stride。

attention 格式差异以及 serving 配置可能带来的 cache dtype 差异，都不只是 permutation
问题。因此本改动不宣称可以把 `amd/GLM-5.1-MXFP4` checkpoint 的所有 weight 完整
零拷贝加载；当前对齐了全部 MXFP4 expert value/scale 和 576-wide cache allocation，
同时保留性能更好的既有 attention 与 router 路径。

## 代码结构

| 文件 | 职责 |
|---|---|
| `config.py` | 模型 profile、公开计算/存储模式和 host 参数验证。 |
| `../common/mx_formats.py` | 可供其他 MoE wrapper 复用的 Torch MXFP4/MXFP8 量化与反量化。 |
| `packing.py` | 按模型处理 attention packing，并提供 native 与 ATOM/AITER-compatible expert packing。 |
| `kernel_common.py` | model-configured kernel 共用的底层 helper。 |
| `kernel_layout.py` | 按模型计算 scratch、peer buffer 和 stage layout。 |
| `shared_reuse_moe_kernel.py` | 针对 GLM-5 性能特化的调度、通信、MLA、routing 和专家计算。 |
| `layer.py` | 调优后 GLM-5 mono-kernel 的稳定 host wrapper。 |
| `indexed_mla_moe_kernel.py` | 由 `LayerConfig` 参数化、便于扩展的 indexed MLA + MoE kernel。 |
| `indexed_layer.py` | 通用 indexed wrapper，以及 GLM-5 compatibility 和 Kimi-K3 MLA adapter。 |
| `kimi_k3.py` | 组合 indexed MLA、latent MoE 与 reduction 的 Kimi-K3 完整层 adapter。 |
| `kda.py` | 生产 shape 的 TP8 KDA attention，以及显式 state-pool API。 |
| `kda_conv.py`、`kda_recurrence.py` | slot-indexed KDA convolution 与融合 recurrence/gated-RMSNorm kernel。 |
| `kimi_k3_attn_res.py` | 融合 Kimi-K3 AttnRes mixing、RMSNorm 与 MXFP8 输入量化。 |
| `router_projection.py`、`mxfp8_linear.py` | 融合 router/top-k，以及可复用的 gfx950 MXFP8 projection 支持。 |
| `kimi_k3_tail.py` | 重叠 routed reduce/RMSNorm、MXFP8 shared/latent projection 与最终 TP reduce。 |
| `router.py`、`router_projection.py` | 可复用的原生 routing 与融合 projection/top-k kernel。 |
| `runtime.py`、`../common/hip_ipc.py` | 共用 symmetric HIP IPC 生命周期和远端 handle 的确定性清理。 |
| `symmetric_allreduce.py`、`torch_fusions.py` | 可 graph capture 的 reduction 与编译后 Torch output helper。 |
| `reference.py` | 独立 Torch 分段与端到端计算。 |
| `native_baseline.py` | 可选的同权重 TileRT 对比适配器。 |
| `tools/benchmark_atom.py` | 使用预选 sparse indices 的 ATOM 原生 GLM-5.1 decoder-layer benchmark。 |
| `tools/kimi_k3_full.py` | Kimi-K3 正确性、profiling 和完整层 benchmark driver。 |
| `../kimi_k3/tools/full_layer.py` | 单次 launch 的 KDA + AttnRes + latent-MoE TP8 正确性/性能工具。 |

kernel 使用 FlyDSL 操作实现 wave reduction、硬件数学指令、mailbox polling、buffer
访问和 MFMA。每个 rank 的 peer payload 先舍入为 BF16，再按 rank 顺序累加，因此所有
rank 得到逐位一致的 hidden state 和 routing 结果。

`SharedReuseMlaMoeLayer` 持有远端 HIP IPC mappings。应在最后一次 rank barrier 后调用
`close()`，也可以把该对象作为 context manager 使用。

## 正确性状态

`w8a8` 和 `w8a16` 均通过 2/4/8 GPU x S=1/2/4 的完整矩阵。九种配置各运行五组变化
输入，并检查：

- 分段输出与独立 Torch 计算一致；
- 所有 rank 的最终输出逐位一致；
- 最终 down projection 和 BF16 peer reduction；
- 输出有限，且 HIP graph replay 稳定。

独立 FP32 端到端检查通过 41 个输入，相对 L2 为 1.49-2.85%。另有四个输入沿用已有的
近似并列 routing 跳过规则，因为 attention 的一个 BF16 ulp 差异改变了专家选择；这些
输入仍通过分段检查和跨 rank 逐位一致检查。现有容差没有放宽。一个 NP2/S4 `w8a16`
intermediate 使用已有的一个 BF16 ulp 上限，而其最终 down/output 仍精确匹配。

S=8 也在 1、2、4、8 GPU 上分别以一组新输入通过 `w8a8` 和 `w8a16` 的完整分段检查。
更大的 peer payload 使用两个 64-lane send batches，仍保持所有 rank 输出逐位一致。
NP4 `w8a16` 的 normalized expert input 有一个元素与独立 reduction 相差一个 BF16 ulp，
处于已有 BF16 handoff 上限内。

新增的 `a16w4` 和 `a8w4` 在单卡 S=1、S=8，以及双卡和八卡 S=8 上通过完整分段
检查。检查覆盖 packed MXFP4 weight、每 1x32 E8M0 scale、A8W4 activation 量化、
最终 BF16 peer reduction，以及所有 rank 最终输出逐位一致。TP8/S8 的独立端到端
相对 L2 分别为 `a16w4` 0.430%、`a8w4` 2.87%。没有放宽现有容差。

ATOM layout 路径还通过了两种 MXFP4 模式的默认布局 TP1/S1、TP1/S8 和 TP8/S8
检查。A8W4 S>1 最初暴露了一个只影响 shared expert 的 scale
关联问题：8 个 routed slots 都正确，只有 shared slot 0 错误。最终实现让 routed
experts 继续使用高性能 lane-group ATOM 直读，仅在一个 MFMA 的不同列代表不同 sample
的 shared expert 上使用 canonical K32 gather。

TP1/S1 下使用相同权重直接对比 TileRT wrapper，结果为：

| 模式 | 最大绝对误差 | 相对 L2 |
|---|---:|---:|
| `w8a8` | 0.1171875 | 2.263% |
| `w8a16` | 0.03125 | 0.357% |

## 性能状态

### MXFP4：FlyDSL mono-kernel 与 ATOM 原生层

MXFP4 主对比运行于 8 x MI355X（gfx950）、TP8，位置 3000、sparse top-2048，
KV cache 为 BF16 `[tokens,576]`。每个 HIP graph 只包含一次 decoder-layer 调用。
预热两次后记录 30 次 replay；每轮取最慢 rank，去掉最快 5 次和最慢 5 次，再对剩余
20 次求平均。

ATOM 把 GLM-5.1 注册为 `GlmMoeDsaForCausalLM`。该生产模型复用
`atom.models.deepseek_v2` 并构造 `DeepseekV2DecoderLayer`，因此 baseline 直接使用
GLM-5.1 config 实例化这一个 GLM 实际选择的 layer class；继承而来的 class 名称不代表
使用了 DeepSeek 模型配置。其原生多算子链路包括 RMSNorm、BF16 MLA projections 与
sparse attention、router、MXFP4 FusedMoE 和 TP collectives。仅跳过生成 sparse top-2048
indices 的 indexer，因为 FlyDSL API 同样把这些 indices 作为输入。版本为 ATOM
`b104cf915aeced8c0c319fe1e0fcf6cf70b323ba`、AITER
`d4e9afc85857e30e03b417606e6f25071ffaaa9a`。

| Batch | FlyDSL mono-kernel | ATOM 原生层 | 加速比 | 延迟降低 |
|---:|---:|---:|---:|---:|
| 1 | 74.77 us | 136.84 us | 1.83x | 45.4% |
| 2 | 73.69 us | 145.43 us | 1.97x | 49.3% |
| 4 | 94.77 us | 144.12 us | 1.52x | 34.2% |
| 8 | 124.66 us | 158.99 us | 1.28x | 21.6% |

所有已测 batch 均未出现相对 ATOM 的性能回退，因此不需要继续拆分
attention、router/MoE 和 collective 排查。这是等价单层工作量的延迟对比，不是纯粹的
fusion-only 对比：expert 与 KV storage 使用对齐的 MXFP4/BF16 格式，但已发布的
`amd/GLM-5.1-MXFP4` ATOM attention weights 为 BF16，而 FlyDSL mono-kernel 保留
block-128 FP8 attention 路径。

### 历史数据与 layout A/B 测量

下表为既有 W8A8 HIP graph 测量：每个 graph 含 128 次 layer launch，预热两次，正式
测量九次，并取各轮最慢 rank 的中位数。硬件为 8 x MI355X（gfx950），位置 3000，
sparse top-2048。

| GPU 数量 | 后端 | S=1 | S=2 | S=4 |
|---:|---|---:|---:|---:|
| 2 | FlyDSL | 33.98 us | 39.80 us | 53.55 us |
| 4 | FlyDSL | 34.30 us | 41.51 us | 54.42 us |
| 8 | FlyDSL | 35.55 us | 42.72 us | 56.32 us |
| 8 | TileRT | 35.85 us | 42.90 us | 55.93 us |

FlyDSL 在已测八卡 S=1、S=2 中更快，在 S=4 中慢 0.7%，所以尚未在所有配置上超过
TileRT。

一次使用 8 层、1 次正式 replay 的 TP1/S1 短 smoke 测量中，`w8a8` 为 33.840 us，
TileRT 为 33.520 us；`w8a16` 为 34.735 us，TileRT 为 32.895 us。这些短运行用于验证
benchmark 路径，不属于可发布的性能数据。

另一组使用 16 层和 3 次正式 replay 的 TP1 测量中，W8A8 的 S=4 为 52.03 us，S=8
为 84.20 us。TileRT 没有 S=8 整层基线。

加入 MXFP4 路径后，在同一进程内使用 16 层、3 次正式 replay 进行 TP1 短测，得到
以下中位数。这些是开发阶段短测，不属于可发布性能数据：

| 模式 | S=1 | S=8 |
|---|---:|---:|
| `w8a8` | 33.38 us | 83.46 us |
| `w8a16` | 34.63 us | 90.13 us |
| `a16w4` | 33.22 us | 78.80 us |
| `a8w4` | 35.21 us | 87.00 us |

`a16w4` 在这组 TP1 短测中最快。`a8w4` 仍需承担每 1x32 activation 量化和 BF16
MFMA staging 的开销，因此没有在两个 sample count 上都超过 `w8a8`。当前已发布
TileRT 对比适配器只接受 `w8a8` 和 `w8a16`，本 harness 中没有两种 MXFP4 模式的
有效同权重 TileRT 基线。

作为 storage-layout 改动的补充证据，ATOM layout 测试在 TP1 下每个 graph 包含
128 次 layer launch，预热两次、正式测量 15 次并取中位数。ATOM expert/cache 测量
前后各测一次 native，以暴露机器漂移；百分比使用两次 native 中位数的均值计算。

| 模式 | Samples | Native expert/scale + split cache | ATOM expert/scale + fused cache | 变化 |
|---|---:|---:|---:|---:|
| `a16w4` | 1 | 31.80-32.33 us | 31.14 us | 快 2.9% |
| `a16w4` | 8 | 74.65-75.04 us | 71.37 us | 快 4.6% |
| `a8w4` | 1 | 32.98-33.02 us | 32.08 us | 快 2.8% |
| `a8w4` | 8 | 81.64-81.80 us | 77.21 us | 快 5.5% |

轻量 TP8 graph benchmark 每个 graph 包含 16 次 launch，并取最慢 rank，也没有出现
回退：

| 模式 | Samples | Native | ATOM expert/cache 默认布局 |
|---|---:|---:|---:|
| `a16w4` | 1 | 35.2 us | 34.8 us |
| `a16w4` | 8 | 87.2 us | 82.1 us |
| `a8w4` | 1 | 36.9 us | 35.4 us |
| `a8w4` | 8 | 92.9 us | 87.5 us |

第一版 ATOM expert layout 使用分散 dword load 和寄存器内转置，A16W4 S=8 曾上升到
约 115 us。把四个 lane groups 直接映射到 ATOM 的四个 K32 tiles 后消除了这项回退，
并保留 ATOM scale storage 与 fused cache。A8W4 多 sample 的 shared-expert 路径使用
上文所述的小范围 canonical-load fallback 保证正确性；最终 TP1/S=8 仍比 native
layout 快 5.5%。

分段 trace 指导了两项保留的调度修改：BF16 packed peer exchange，以及每个 router
CTA 只处理一个 sample。此前 S=4 调度中，最后一个插桩 CTA 到达 attention 发布、
router 发布、routed up/gate 发布和 down 完成的时间分别为 30.23、35.12、50.12、
57.52 us；修改后为 27.66、30.65、45.73、52.97 us。对比图位于
`/root/glm5-perf-results/s4-segment-milestones.png`。

## 复现

使用现有 FlyDSL compiler build 和本 worktree：

```bash
cd /root/FlyDSL-glm5-mxfp4-atom-layout
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-mxfp4-atom-layout
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode a16w4 --npes 8 --samples 1 2 4 8 \
  --layers 1 --repeats 30 --trim 5

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark_atom.py \
  --npes 8 --samples 1 2 4 8 --layers 1 --repeats 30 --trim 5
```

正确性命令应替换其他模式和 peer count 重复运行。

ATOM benchmark 默认使用 `/root/ATOM` 和 `/root/aiter`；如路径不同，可通过
`--atom-root` 和 `--aiter-root` 覆盖。GPU 任务应串行执行。

FlyDSL benchmark 可添加 `--trace --layers 16 --trace-dir <directory>` 记录分段时间戳，
然后查看某个 rank：

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

trace 插桩会等待内存操作完成并改变调度。延迟比较应使用未插桩的 graph 测量。

## Kimi-K3 完整 MLA + latent-MoE 层

`KimiK3MlaLayer` 继续作为可复用的 full-attention 组件。新增的
`KimiK3MlaMoeLayer` 实现了 Kimi-K3 生产 TP8 decoder layer 的完整数据路径：

- hidden size 7168、Q-LoRA rank 1536、KV-LoRA rank 512；
- 每卡 12 个 head，non-positional / RoPE / value 维度分别为 128 / 64 / 128；
- BF16 attention 权重，以及 sigmoid attention output gate；
- attention 前和 MoE 前的 12-layer AttnRes source mixing；
- BF16 router 投影、FP32 sigmoid/correction-bias 选路、896 experts 和归一化 top-16；
- replicated MXFP8 7168→3584 latent projection；
- FlyDSL device-side sorting，以及带 SiTU 的两阶段 A16W4/MXFP4 routed experts；
- TP8-local MXFP8 shared experts、latent RMSNorm 和 rank-local MXFP8 3584→896 tail；
- latent 空间的一次 TP reduce，以及 residual 更新前的最终 TP reduce。

A16W4 launcher 和 Kimi-K3 tuned 配置现在位于
`kernels/moe/moe_2stage_a16wmix/host.py`；测试改为导入该生产模块，不再自行持有
host 实现。

正确性在 8 x MI355X（gfx950）上验证了 layer 0 的 S=1/4/8，并以 S=4 验证了
layer 1 和 layer 12，覆盖非 block-write 与新 block-write 两类 AttnRes 分支。所有输出
均为有限值，且八个 rank 的结果逐位一致。最终优化版 S=4/S=8 使用实现自身的
post-attention 状态时，top-16 mismatch 均为 0，routed-MoE relative L2 为
0.549%/0.538-0.551%，完整输出 relative L2 约为 0.469%/0.466-0.474%，KV-cache
relative L2 约为 1e-8。额外 5 次全新进程的 S=8 检查都保持 rank 间逐位一致，且未再
复现此前单次较高误差样本。layer 0/1/12 的 HIP graph capture 与 replay 也均成功。

与现有 GLM-5 测试相同，独立端到端比较会记录、但不会因合法 MLA rounding 差异导致的
近似并列 synthetic route 翻转而失败。隔离的 MoE 检查把实现的 post-attention tensor
传给独立 MoE reference，因此可以把真实 router/expert 回归与该 synthetic 边界效应
区分开。

MLA 组件保留原有两项调度优化：S >= 2 时，两个 local head group 复用同一份 64-key
KV tile 和一次 16-column score MFMA；output gate 则融合到每个 W_UV producer。
attention-only TP8 在 S=1、S=4、S=8 的提升仍分别为 0.6%、2.3%、12.0%。

### Synthetic MLA 调优性能

原始 synthetic `layer_idx=0` MLA harness 使用 TP8、position 3000、每个 HIP graph 16 次 layer launch、2 次
eager forward 预热、2 次 graph replay 预热、100 次正式 replay，并取最慢 rank 的中位
延迟。测试前已确认 8 张 GPU 没有其他进程。最终路径使用融合 BF16 router/top-16 与
MXFP8 latent/shared projection、`bm16` MXFP4 routed-MoE tile、融合通信/RMSNorm/tail，
以及可 graph capture 的 symmetric peer mailbox。

| 版本 | S=4 | S=8 |
|---|---:|---:|
| 第一版正确的串行完整层 | 383.85 us | 404.81 us |
| 深度优化前的公共模块源码 | 258.7604 us | 282.4305 us |
| 融合 router projection 前 | 204.4796 us | 228.6948 us |
| MXFP8/tail 集成前的 fused-router 源码 | 194.9370 us | 221.5046 us |
| 最终 MXFP8/overlap 源码 | **109.4035 us** | **124.0237 us** |
| 相对公共模块源码的加速 | **2.37x** | **2.28x** |

早期受控 NCCL 与 symmetric reduce A/B 中，S=4 仅从 258.9820 us 变为
256.3328 us，S=8 仅从 281.1871 us 变为 279.5571 us，差异约 0.6-1.0%。因此只替换
通信并不足够。最终一轮同时处理了 launch 开销、occupancy 不足、重复访存与 overlap：

- routed experts 使用现有调优后的 `bm16` MXFP4 GEMM 路径，与 ATOM 的低 token 选择
  对齐。该阶段把 S=4/S=8 从 235.29/255.39 us 降到 226.59/247.25 us。
- `mxfp8_linear.py` 沿用仓库已有的 gfx950 scaled-MFMA preshuffle 布局，并针对
  M=1/2/4/8 特化调度。post-AttnRes 只量化一次 activation，latent-down 与 shared
  up/gate 共同消费这份 packed 输入。
- `router_projection.py` 把 BF16 router projection/top-16 与两路 MXFP8 projection 放在
  同一个 kernel。每个 512-thread CTA 使用 4 个 projection wave，增加可独立调度的 CTA
  数；隔离的 latent/shared 阶段从约 24.9 us 降到 16.2 us。按 sample 数调整 CTA
  规模后，完整融合 kernel 在 S=4/S=8 分别为 22.44/23.36 us。
- K3 的 `S <= 8`、top-16 约束意味着每个 active expert 只需要一个 padded 16-row tile。
  LDS atomic histogram 与每条 route 的 atomic ticket 取代重复扫描，使 S=8 整层减少约
  8 us。
- `kimi_k3_tail.py` 把 routed TP reduce/RMSNorm 与独立的 shared-down 计算重叠。shared
  partial 提前发给对应 output owner；owner 随后叠加本 rank 的 latent-up shard，再广播
  最终值。两路 projection、两段通信、RMSNorm、累加与 residual update 合并后，该 kernel
  在 S=4/S=8 分别为 14.97/16.83 us。shared-down 与 latent-up 的 MXFP8 weight/scale
  现在也使用与其他 MXFP8 projection 相同的 ATOM/AITER preshuffle，其中 scale 按 256 行
  padding。相对紧邻的上一版，完整层 S=4/S=8 从 111.3647/126.0886 us 降到
  109.4035/124.0237 us。
- 独立 AttnRes kernel 在一次 launch 内完成混合，并直接量化 post-attention 输出供 MXFP8
  消费。post-AttnRes 在 S=4/S=8 分别为 8.22/8.51 us。

indexed attention 路径也支持通过 `--kv-cache-layout atom` 使用 ATOM-compatible 的融合
BF16 `[tokens,576]` KV/PE cache。该路径与 split cache 数值一致，实测差异约 0.1 us，
因此目前仍保留 split layout 为默认值。

隔离的 7168→3584 projection 使用含 16 次 launch 的 HIP graph 测量，MXFP8 kernel
在 M=4/M=8 分别为 7.05/7.75 us；对应 BF16 `torch.mm` 为 11.86/12.09 us，因此特化
MXFP8 GEMM 分别快 1.68x/1.56x，同时相对反量化 MXFP8 matmul 的 relative L2 约为
`2e-4`。

router projection 和 correction bias 也已改成 BF16，与 ATOM 的生产 gate 契约一致；
FP32 只用于 sigmoid、比较和 route normalization。该 dtype 修正没有明显改变延迟，
但消除了实现对比中的语义差异。

按要求与 ATOM 原始实现进行对比时，使用 ATOM commit `3cea04f45`，直接实例化其生产
`atom.models.kimi_k3.KimiDecoderLayer`，覆盖 AttnRes、MLA、routing、routed/shared
MoE、latent transforms、TP reductions、dual streams 和 HIP graph replay。

| Batch | FlyDSL 完整层 | ATOM 完整层 | 加速比 | 延迟降低 |
|---:|---:|---:|---:|---:|
| 4 | **109.4035 us** | 225.6496 us | **2.063x** | **51.52%** |
| 8 | **124.0237 us** | 256.5222 us | **2.068x** | **51.65%** |

这些数据是 synthetic MLA geometry 的实测端到端时间，不是生产 layer 0 的结果；
attention 工作量也并不完全相同：ATOM
的 `KimiFullAttention` 扫描 dense 3001-token KV context，而 FlyDSL 在 position 3000
消费调用方给出的 top-2048 KV indices。MoE 与 hidden/model shapes、TP8 拓扑、graph
长度、预热次数、正式重复次数和 critical-rank 取值规则一致。因此该表可作为直接的整层
实现基线，但 delta 不能解释为同 attention 工作量下的归一化 kernel 性能差异。

### 生产 MLA 与 KDA 层族

生产 Kimi-K3 在 zero-based layer 3 及之后每隔 4 层使用 MLA，其余大多数层使用 KDA。
Layer 0 是 KDA + dense FFN，本轮按要求不处理。当前 KDA 路径已经覆盖 BF16 fused input
projection、低秩 `f_a -> f_b` gate、slot-indexed causal convolution、FP32 recurrent
delta-rule state update、gated per-head RMSNorm、BF16 output projection、TP8 reduction、
前后两段 AttnRes，以及已有的 latent MXFP4 MoE。

`f_b` projection 已融合进 convolution/recurrence kernel：每个 per-head CTA 从共享的
128-wide `f_a` 输入计算自己的 128 个 gate channel，并立刻消费，去掉一个独立的低
token GEMM，同时保持显式、graph-safe 的 state-pool API。

KDA output projection 现在也带有专用 TP8 epilogue。每个 GEMM CTA 直接从 LDS 把
BF16 output tile 发布到已有的 tagged symmetric peer mailbox，等待对应 epoch tag 后，
把归约结果直接写入 attention output。这样同时去掉了独立 all-reduce launch 和中间
`partial` 的写回/重读。S=1-4 使用 16x64、4-wave tile；S=8 使用 32x64、8-wave tile，
使每个 wave 正好负责一个 TP peer。

下表使用相同生产层索引、TP8 拓扑、16 层 HIP graph、预热、30 次重复和 critical-rank
中位数进行配对。ATOM harness 现在会初始化生产 AttnRes block state，不再对未初始化
storage 计时。

| 层族 / 代表层 | Batch | FlyDSL | ATOM | 加速比 | 延迟降低 |
|---|---:|---:|---:|---:|---:|
| MLA + latent MoE，layer 3 | 4 | **116.1023 us** | 222.5597 us | **1.917x** | **47.83%** |
| MLA + latent MoE，layer 3 | 8 | **127.9274 us** | 253.9299 us | **1.985x** | **49.62%** |
| KDA + latent MoE，layer 1 | 4 | **115.3998 us** | 197.2968 us | **1.710x** | **41.51%** |
| KDA + latent MoE，layer 1 | 8 | **128.8000 us** | 227.3771 us | **1.765x** | **43.35%** |

生产 MLA 的 S=8 已约为 2x，KDA 完整层仍未达到 2x。保留的融合 KDA
convolution/recurrence/RMSNorm core 在 S=4/8 相对 ATOM 三 kernel 链约为
1.95-1.98x。融合后的 output projection + TP reduction 在 S=4/S=8 分别约为
9.7/12.7 us，而之前的双 launch 路径约为 14.1/14.5 us。完整 KDA 达到 2x 需要降至
S=4 98.65 us、S=8 113.69 us，因此仍需再减少 16.75/15.11 us。剩余差距横跨 input
projection、前后两段 AttnRes、router projection 和 routed expert kernels，已经无法靠
微调一个小 kernel 补齐。更快但会放大 recurrent-state 误差的 split-K input projection，
以及让 post-AttnRes 直接消费 peer mailbox 但造成延迟回退的方案，均未保留。

按要求不计 layer 0 dense FFN 后，生产层组合为 24 个 MLA + latent-MoE 层和 68 个
KDA + latent-MoE 层。用上表配对的代表层延迟加权，92 层 decoder core 估算为：S=4
约 **1.764x**，S=8 约 **1.822x**。这不是完整 93 层端到端模型数据；layer 0、
embedding、sampling、framework scheduling 以及其他非层开销尚未计入，因此完整模型
实测会略低。

### 为什么 K3 的绝对耗时仍明显高于 GLM kernel

上文 GLM-5 W8A8 S=4 为 56.217 us，而 K3 A16W4 完整层为 109.4035 us，但不能把
两者当成同工作量的直接优化目标。K3 hidden size 为 7168 而不是 6144，routed expert
为 896/top-16/intermediate 384 而不是 256/top-8/intermediate 256，每卡 attention
head 为 12 而不是 8。仅 router projection 的规模就约大 4.08 倍：
`(7168 * 896) / (6144 * 256)`。

K3 还额外执行 AttnRes mixing、replicated 7168→3584 latent projection、shared
experts、latent RMSNorm、rank-local 3584→896 tail，以及两次 MoE TP reduction。最终
K3 路径仍包含 6 个应用 kernel，而 GLM 快路径把大部分工作放在一个 persistent
monokernel 内。本次保留了 GLM 中有效的机制：persistent attention、owner-reduce/
broadcast 通信、低 token MXFP4 tile、直接生成 routing metadata，以及显式计算/通信
overlap。

最终单层 kernel breakdown 如下：

| 阶段 | S=4 | S=8 |
|---|---:|---:|
| Persistent MLA | 34.07 us | 43.00 us |
| Router + latent/shared projection | 22.44 us | 23.36 us |
| MXFP4 routed GEMM1 | 21.07 us | 22.41 us |
| 融合 shared/latent tail + TP communication | 14.97 us | 16.83 us |
| MXFP4 routed GEMM2 | 9.03 us | 10.15 us |
| Post-attention AttnRes | 8.22 us | 8.51 us |

各阶段相加已接近无插桩端到端延迟，因此剩余时间主要受计算量限制，而不是隐藏的 framework
或 collective 空泡。曾测试用独立 HIP stream 重叠 shared 与 routed 分支，但小 kernel
争抢计算资源并增加跨 stream 同步，因此未保留。最终保留的 overlap 位于 fused tail
内部：shared 访存和远端发送在 routed reduce/RMSNorm 与 latent-up 依赖完成期间推进。
也测试过把 selector CTA 放到 projection CTA 前面；修正区间条件后结果正确，但 S=8
回退到 127.12 us，因此保留 selector-last 调度。

完整 MLA + MoE 正确性与性能可用以下命令复现：

```bash
cd /root/FlyDSL-kimi-k3
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-kimi-k3

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 4 --layer-idx 3 --check \
  --bench --kernel-profile --layers 16 --repeats 100 \
  --output /root/kimi-k3-perf-results/full-moe/final-s4.json

/opt/venv/bin/python kernels/mla_moe_layer/tools/kimi_k3_full.py \
  --npes 8 --samples 8 --layer-idx 3 --check \
  --bench --kernel-profile --layers 16 --repeats 100 \
  --output /root/kimi-k3-perf-results/full-moe/final-s8.json

python -m kernels.kimi_k3.tools.full_layer \
  --npes 8 --samples 4 --layer-idx 1 --check --bench \
  --layers 16 --repeats 30 \
  --output /root/kimi-k3-perf-results/full-moe/flydsl-kda-production-layer1-s4.json
```

可使用 `--eager-attn-res`、`--eager-router` 或 `--eager-shared-experts` 做受控的优化
A/B；`--profile` 可报告 eager GPU event 中位分段时间。默认的
`--reduce-backend symmetric` 可切换为 `nccl` 做通信 A/B。两个完整层 harness 都可
添加 `--kernel-profile`，记录一次 rank-0 graph replay 的 GPU kernel 分解。最终延迟
仍应以未插桩的 HIP graph replay 为准。

ATOM baseline 从独立 checkout 复现，直接使用其原始层实现。本地 AITER JIT build 需要
composable-kernel submodule，以及与这里的 AITER core ABI 匹配的 `pybind11==3.0.1`：

```bash
git -C /root/ATOM-k3-baseline checkout 3cea04f45
git -C /root/aiter submodule update --init --recursive -- 3rdparty/composable_kernel
/opt/venv/bin/python -m pip install --upgrade --target /tmp/atom-k3-deps pybind11==3.0.1

cd /root/FlyDSL-glm5-mxfp4-atom-layout
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 4 --output /root/kimi-k3-perf-results/full-moe/atom-s4.json
/opt/venv/bin/python kernels/mla_moe_layer/tools/atom_kimi_k3_full.py \
  --samples 8 --output /root/kimi-k3-perf-results/full-moe/atom-s8.json
```
