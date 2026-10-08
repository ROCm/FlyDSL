# Kimi-K3 one-kernel 当前版本

本分支 `codex/kimi-k3_one_kernel` 已迁入 session
`01a10f4c-f6ac-7563-ad99-faf460c01281` 最终保留的
`opt254_swizzled_input_lds_s4`。145 个 kernel Python 文件与该实验快照逐字节一致，
包含完整 kernel、host wrapper、权重打包、编译期配置和 reference 修正。
Opt293/294 未获得稳定性能收益，未纳入此版本。

入口为 [KimiK3MonoKernel](../../kernels/kimi_k3_monokernel/op.py)，
支持 TP8、batch 1–8、seq 1–4 的完整 KDA MoE layer 单次 launch。
`samples = batch * seq`；`seq_len` 指定每个独立状态链的连续 token 数，
`seq > 1` 使用 MTP 状态快照。`KimiK3CompileConfig(path="auto")`
在编译期隔离 small_batch/general 路径，默认选择已验证配置。
性能与完整精度覆盖针对 layer 1；layer 0 dense FFN 不在范围内。

## 已验证性能

MI355X 节点46、TP8、完整 layer 1、无插桩 GPU-event 计时，graph16/repeats50；
每个 repeat 取最慢 rank，再取中位数。配对 seeds 为 1234/2025/3141。

| 形状 | 保留耗时 μs | 依据 |
|---|---:|---|
| B1/S1 | 71.826909 | 沿用与 Opt198/229 完全相同的二进制及已验证计时 |
| B1/S4 | 114.204876 | Opt254 三 seed 独立确认；首轮为 113.974907 |

Opt254 相对同期 Opt229 的 S4 首轮及独立确认共六组配对均有正收益，
但幅度很小（0.006–0.855 μs）。不同轮次的绝对数值不能作为配对收益。
B1/S1 60 μs 阶段目标以及原 40/70 μs 目标均未达到。

32 个形状全部编译，31 个非 B1/S4 二进制与 Opt229 一致。
结合继承的原始记录和 Opt254 的六 seed 严格回放，覆盖
192 cases、1536 rank checks、26496 records；精度阈值未放宽。
B1/S1/B1/S4 分别使用 111/123 VGPR、24064/67456 B LDS，
private/VGPR 显存 spill 为零，八卡完整网格驻留检查通过。

## 运行与复核

在配置好 gfx950 FlyDSL/PyTorch 的八卡环境、确认设备空闲并完成对应二进制的
资源与完整网格驻留检查后，从仓库根目录运行：

```bash
PYTHONPATH="$PWD/experiments/kimi_one_kernel/opt254${PYTHONPATH:+:$PYTHONPATH}" \
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --batch 1 --seq 1 --layer-idx 1 --seed 1234 --check --full-replay-check \
  --bench --layers 16 --repeats 50 --output /tmp/kimi_b1_s1.json
```

改为 `--seq 4` 可运行 B1/S4。严格回放使用此目录中封存的 `full_replay.py`；
必须保留上述 PYTHONPATH，使入口加载与历史验证相同的 oracle。
其他五个精度种子为 2025、3141、4242、5678、9999。
编译和驻留工具位于 [opt254](../../experiments/kimi_one_kernel/opt254/)，例如：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
FLYDSL_RUNTIME_CACHE_DIR=/tmp/kimi_opt254_b1_s1_cache \
python experiments/kimi_one_kernel/opt254/compile_shape.py \
  --batch 1 --seq 1 --out /tmp/kimi_opt254_b1_s1_resources
python experiments/kimi_one_kernel/opt254/check_shape_occupancy.py \
  /tmp/kimi_opt254_b1_s1_resources
```

每个编译形状使用独立的空缓存目录。重新编译后的资源或二进制可能受工具链影响，
历史驻留证据不能代替新二进制的检查。

本次提交迁移未新增 GPU 测量；CPU 复核检查源文件身份、32 个配置与封存编译记录一致、
旧归档补丁可还原，以及历史原始精度/计时/二进制证据。
仓库内源码和归档的复核命令不需要 GPU 或 FlyDSL：

```bash
python3 experiments/kimi_one_kernel/verify_archive.py
```

[源码 manifest](../../experiments/kimi_one_kernel/opt254/source_manifest.json)、
[选择记录](../../experiments/kimi_one_kernel/opt254/CURRENT_SELECTION.json)、
[完整功能覆盖](../../experiments/kimi_one_kernel/opt254/opt254_inheritance_validation.json)及
[性能验证](../../experiments/kimi_one_kernel/opt254/opt254_validation.json)保留原始记录。
其中 `repository_checkout_changed: false` 等字段描述实验采用时的状态，
并非本次迁入 Git 后的状态。

原始二进制、逐 rank 回放、逐次 event 计时和压缩包保留在工作区的
`../results/kimi_shapes_20261006/`（相对于此仓库根目录）；46 容器路径为
`/tmp/flydsl-kimi-shapes-20261006/`。
压缩包摘要见上述验证记录，Git 中仅保留源码、工具及可审阅的证据摘要。

[2026-09-29 历史归档](archive_20260929.md)中的 92 个补丁仍以
`8137256d25ccea61eeca4f43d26fc3aad280cb1d` 为基底，不能直接叠加到 Opt254。
归档校验器通过 Git 中该提交还原旧基底，同时校验当前 Opt254 源码。
