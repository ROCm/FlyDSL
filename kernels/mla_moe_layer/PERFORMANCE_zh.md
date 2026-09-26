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
| KV cache | 每个 token 为 `[k_c(512), k_pe(64)]` | 单个连续 BF16 `[tokens, 576]` tensor | shape/stride 一致；ATOM benchmark 使用 FP8 而非 BF16 |

expert value 和 scale packer 已针对 GLM-5 up/gate、down 两种形状，与当前 AITER 的
`shuffle_weight`、`shuffle_scale` 做过逐字节比较，结果完全一致。norm vectors 和
routing bias 原本就是普通连续 tensor。ATOM 默认的 `ATOM_MOE_GU_ITLV=0` 也与本
kernel 的 gate-then-up 行顺序一致。

保留 `--router-weight-layout atom` 用于零拷贝实验，但不设为默认：在其他存储均采用
ATOM-compatible 的 A16W4 下，TP1 S=1 从 31.19 降到 31.08 us/layer，略有提升；但
S=8 从 70.80 上升到 72.47 us/layer，回退 2.4%。ATOM 的独立 GEMM 可以在内部对
row-major router 做 tiling，而 mono-kernel 的直接 MFMA load 会跨越 6144-element
row stride。

attention 与 cache dtype 的差异属于格式变化，不只是 permutation 问题。因此本改动
不宣称可以把 `amd/GLM-5.1-MXFP4` checkpoint 的所有 weight 完整零拷贝加载；当前
对齐了全部 MXFP4 expert value/scale 和 576-wide cache allocation，同时保留性能更好
的既有 attention 与 router 路径。

## 代码结构

| 文件 | 职责 |
|---|---|
| `config.py` | 固定维度、公开计算模式和 host 参数验证。 |
| `../common/mx_formats.py` | 可供其他 MoE wrapper 复用的 Torch MXFP4/MXFP8 量化与反量化。 |
| `packing.py` | FP8、BF16 和 MXFP4 matrix 的 MFMA weight packing。 |
| `runtime.py` | 自有 symmetric HIP IPC buffer，以及远端 handle 的确定性清理。 |
| `shared_reuse_moe_kernel.py` | FlyDSL kernel 调度、通信、MLA、routing 和专家计算。 |
| `layer.py` | 公开 host wrapper、scratch 分配、启动参数、trace 和生命周期。 |
| `reference.py` | 独立 Torch 分段与端到端计算。 |
| `native_baseline.py` | 可选的同权重 TileRT 对比适配器。 |

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

ATOM layout 测试在 TP1 下每个 graph 包含 128 次 layer launch，预热两次、正式测量
15 次并取中位数。ATOM expert/cache 测量前后各测一次 native，以暴露机器漂移；百分比
使用两次 native 中位数的均值计算。

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
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-mxfp4-atom-layout:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode a16w4 --npes 1 --samples 1 8 \
  --layers 128 --repeats 15
```

正确性命令应替换其他模式和 peer count 重复运行。GPU 任务应串行执行。

如需与已发布实现直接比较，保留 `/root/tilert_pkg` 在 `PYTHONPATH` 中，并把
`--backend flydsl` 改为 `--backend tilert`。原生 wrapper 只支持 `w8a8`/`w8a16`、
1 或 8 peers，以及 sample count 1/2/4。S=8 和 MXFP4 模式是本 harness 中的
FlyDSL 独有扩展。

FlyDSL benchmark 可添加 `--trace --layers 16 --trace-dir <directory>` 记录分段时间戳，
然后查看某个 rank：

```bash
/opt/venv/bin/python kernels/mla_moe_layer/tools/profile_summary.py \
  <directory>/w8a8-s4/rank0/trace.pt
```

trace 插桩会等待内存操作完成并改变调度。延迟比较应使用未插桩的 graph 测量。
