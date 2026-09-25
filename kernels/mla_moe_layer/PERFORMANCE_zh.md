# Shared/reuse MLA + MoE kernel

[English](PERFORMANCE.md)

本目录包含固定 GLM-5 MLA + MoE 分片，以及 Kimi-K3 full-attention MLA 分片的
FlyDSL 实现。生产路径通过 `SharedReuseMlaMoeLayer` 和 `KimiK3MlaLayer` 提供，均由
`build_shared_reuse_kernel` 生成。这些路径不导入 TileRT kernel 主体，也不嵌入汇编。
原有的汇编捕获、改写和启动桥接代码已全部删除。

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

分段 trace 指导了两项保留的调度修改：BF16 packed peer exchange，以及每个 router
CTA 只处理一个 sample。此前 S=4 调度中，最后一个插桩 CTA 到达 attention 发布、
router 发布、routed up/gate 发布和 down 完成的时间分别为 30.23、35.12、50.12、
57.52 us；修改后为 27.66、30.65、45.73、52.97 us。对比图位于
`/root/glm5-perf-results/s4-segment-milestones.png`。

## 复现

使用现有 FlyDSL compiler build 和本 worktree：

```bash
cd /root/FlyDSL-glm5-perf
export PYTHONPATH=/root/FlyDSL/build-fly/python_packages:/root/FlyDSL-glm5-perf:/root/tilert_pkg
export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --npes 8 -S 8 --pos 3000 --iters 1 --moe-mode a8w4

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --moe-mode a16w4 --npes 1 --samples 1 8 \
  --layers 16 --repeats 3
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

## Kimi-K3 full-attention MLA

Kimi-K3 profile 实现了 TP8 ROCm 模型路径使用的 full-attention MLA 分片：

- hidden size 7168、Q-LoRA rank 1536、KV-LoRA rank 512；
- 每卡 12 个 head，non-positional / RoPE / value 维度分别为 128 / 64 / 128；
- BF16 attention 权重，以及 sigmoid attention output gate；
- 输入由调用方预先归一化，输出仅包含完成 TP reduce 的 MLA projection，与
  `KimiMLAAttention` 的接口边界一致。

Kimi-K3 路径有意止于该边界，不宣称是完整 Kimi-K3 decoder layer。模型的 12 层
attention-residual block 和 latent-MoE 尾部仍在 kernel 外部；具体而言，7168→3584
routed transform、896 experts/top-16 MXFP4 MoE、shared SiTU MLP、latent
normalization/up-projection 和最终 residual 均不属于 `KimiK3MlaLayer`。

正确性在 8 x MI355X（gfx950）上完成了 TP1 S=1/4/8 和生产拓扑 TP8 S=1/8 验证。
每个 attention 中间量均与独立 Torch 计算比较，同时检查新增 KV/PE cache 行，并确认
TP8 所有 rank 的最终输出逐位一致；64-layer HIP graph 也完成了重复 replay 验证。
benchmark 现在会在独立 reference launch 与 graph capture 之间推进 mailbox epoch；
复用该 tag 会使不同 rank 读取不同代的 mailbox 数据并造成死锁。

主要优化是在 S >= 2 时，让两个 local head group 复用同一份 64-key KV tile 和一次
16-column score MFMA，避免 heads 8-11 的第二个 CTA 重复读取 Q/KV 并重算 score。
S=1 仍保留两个 CTA，因为并行执行略快。output gate 也下推并融合进各 W_UV producer，
使每个输出元素只计算一次 sigmoid，不再由全部 224 个 W_o CTA 重复计算。

下表是在 position 3000、每个 HIP graph 64 次 launch、7 次正式 replay 条件下得到的
每个 MLA 分片中位延迟。“Baseline”是两项调度优化之前、每个 split 使用两个 CTA 的
正确实现。

| Peer 数 | 版本 | S=1 | S=4 | S=8 |
|---:|---|---:|---:|---:|
| 1 | Baseline | 24.17 us | 32.16 us | 42.99 us |
| 1 | Optimized | 23.60 us | 30.10 us | 35.78 us |
| 1 | 提升 | 2.3% | 6.4% | 16.8% |
| 8 | Baseline | 26.44 us | 35.53 us | 49.77 us |
| 8 | Optimized | 26.29 us | 34.70 us | 43.80 us |
| 8 | 提升 | 0.6% | 2.3% | 12.0% |

插桩后的 TP1/S8 中，最后一个 split、W_UV、W_o 的完成时刻从
34.2/39.7/42.2 us 提前到 28.6/33.7/36.1 us。插桩自身会增加开销，因此这些时间用于
解释关键路径变化，不是上表的最终延迟。

Kimi-K3 的正确性与性能可用以下命令复现：

```bash
/opt/venv/bin/python tests/kernels/test_shared_reuse_mla_moe_layer.py \
  --model kimi_k3 --npes 8 -S 8 --pos 100 --iters 1

/opt/venv/bin/python kernels/mla_moe_layer/tools/benchmark.py \
  --backend flydsl --model kimi_k3 --npes 8 --samples 1 4 8 \
  --layers 64 --repeats 7 --pos 3000
```
