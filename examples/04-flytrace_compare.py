# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Validate static wave tracing; publish one report, three ISAs and one timeline."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
from contextlib import nullcontext
from pathlib import Path


def fingerprint(root):
    files = [
        "examples/04-flytrace_gemm.py",
        "python/flydsl/extension/_flytrace.py",
        "python/flydsl/extension/flytrace.py",
        "python/flydsl/compiler/kernel_function.py",
        "python/flydsl/compiler/jit_function.py",
        "python/flydsl/compiler/jit_executor.py",
        "include/flydsl/Dialect/Fly/IR/FlyOps.td",
    ]
    return hashlib.sha256(b"".join((root / f).read_bytes() for f in files)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--all-blocks", action="store_true")
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()
    if args.batches <= 0 or args.iterations <= 0:
        parser.error("batches and iterations must be positive")
    root = Path(__file__).resolve().parents[1]
    work = root / "build-fly/flytrace/static" / ("all" if args.all_blocks else args.default_device)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(FLYDSL_RUNTIME_ENABLE_CACHE="0", FLYDSL_DUMP_IR="1")

    import torch

    import flydsl.compiler as flyc
    from flydsl.extension import flytrace
    from tests.utils import shuffle_weight

    spec = importlib.util.spec_from_file_location("flytrace_gemm", root / "examples/04-flytrace_gemm.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    results, functions, captures = {}, {}, {}
    modes = ("off", "phases", "tiles")
    with torch.device(args.default_device):
        torch.manual_seed(0)
        a = torch.randn(demo.M, demo.K, dtype=torch.float16, device="cuda")
        b = torch.randn(demo.N, demo.K, dtype=torch.float16, device="cuda")
        c = torch.empty((demo.M, demo.N), dtype=torch.float16, device="cuda")
        shuffled = shuffle_weight(b, layout=(16, 16))
        ta = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=1, divisibility=16)
        tc = flyc.from_dlpack(c).mark_layout_dynamic(leading_dim=1, divisibility=16)
        stream = torch.cuda.current_stream()
        expected = (a @ b.T).float()
        call_args = (ta, shuffled, tc, stream)

        def verify(mode):
            torch.cuda.synchronize()
            assert torch.isfinite(c).all(), f"{mode}: non-finite GEMM output"
            assert ((c.float() - expected).abs() <= 1e-3 + 1e-3 * expected.abs()).all(), f"{mode}: incorrect GEMM"
            if mode == "off":
                return dict(waves=0, records=0, bytes=0)
            capture = captures[mode]
            waves = capture.decode()
            assert len(waves) == (4096 if args.all_blocks else 4)
            for wave in waves:
                assert len(wave["events"]) == (70 if mode == "tiles" else 8)
                if mode == "tiles":
                    assert [e["payload"] for e in wave["events"] if e["name"] == "k_tile"] == list(range(64))
            stats = capture.export(work / mode / "timeline.json")
            stats["bytes"] = sum(spec["words"] * 4 for spec, _ in capture._records.values())
            return stats

        for mode in modes:
            folder = work / mode
            folder.mkdir(exist_ok=True)
            os.environ["FLYDSL_DUMP_DIR"] = str(folder / "ir")
            # This host context is all the capture setup. Kernel/JIT signatures,
            # tensor allocation and loop state need no trace-specific arguments.
            capture = (
                nullcontext()
                if mode == "off"
                else flytrace.capture(
                    exclude=("k_tile",) if mode == "phases" else ("mainloop", "drain"),
                    **({"block": None} if args.all_blocks else {}),
                )
            )
            captures[mode] = capture
            with capture:
                functions[mode] = flyc.compile(demo.preshuffle_gemm, *call_args)
                functions[mode](*call_args)
            stats = verify(mode)
            isa_paths = list((folder / "ir").rglob("*_final_isa.s"))
            assert len(isa_paths) == 1
            shutil.copyfile(isa_paths[0], folder / "final-isa.s")
            isa = isa_paths[0].read_text()
            fields = {
                "vgpr": r"\.num_vgpr, (\d+)",
                "agpr": r"\.num_agpr, (\d+)",
                "sgpr": r"\.sgpr_count:\s+(\d+)",
                "scratch": r"\.private_segment_fixed_size:\s+(\d+)",
            }
            resources = {k: int(re.search(pattern, isa).group(1)) for k, pattern in fields.items()}
            resources.update(
                clock_sites=len(re.findall(r"^\s+s_memrealtime\s", isa, re.M)),
                atomic_sites=len(re.findall(r"^\s+(?:global|buffer|flat)_atomic_", isa, re.M)),
                waterfalls=len(re.findall(r"^\s+s_cbranch_execnz\s", isa, re.M)),
                mfma=len(re.findall(r"^\s+v_mfma_", isa, re.M)),
            )
            assert not resources["scratch"] and not resources["atomic_sites"] and not resources["waterfalls"]
            assert resources["mfma"] == 256
            if mode == "off":
                assert resources["clock_sites"] == 0
                assert functions[mode]._keepalive._trace_spec is None
            results[mode] = dict(correct=True, trace=stats, isa=resources, samples_us=[])
            print(f"{mode}: correctness/trace PASS, {resources}, {stats}", flush=True)

        for mode in modes:
            with captures[mode]:
                for _ in range(20):
                    functions[mode](*call_args)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for batch in range(args.batches):
            order = modes[batch % 3 :] + modes[: batch % 3]
            for mode in order:
                with captures[mode]:
                    for _ in range(5):
                        functions[mode](*call_args)
                    start.record(stream)
                    for _ in range(args.iterations):
                        functions[mode](*call_args)
                    end.record(stream)
                    end.synchronize()
                    us = start.elapsed_time(end) * 1000 / args.iterations
                results[mode]["samples_us"].append(us)
                print(f"batch {batch + 1} {mode}: {us:.3f} us", flush=True)
        for mode in modes:
            with captures[mode]:
                functions[mode](*call_args)
            results[mode]["trace"] = verify(mode)
            results[mode]["median_us"] = statistics.median(results[mode]["samples_us"])

    data = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        default_device=args.default_device,
        all_blocks=args.all_blocks,
        batches=args.batches,
        iterations=args.iterations,
        fingerprint=fingerprint(root),
        results=results,
    )
    (work / "measurements.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.batches >= 5:
        if not args.all_blocks and args.default_device == "cpu":
            publish(root, work, data)
        elif args.all_blocks:
            sampled = root / "build-fly/flytrace/static/cpu"
            if (sampled / "measurements.json").exists():
                prior = json.loads((sampled / "measurements.json").read_text())
                if prior["fingerprint"] == data["fingerprint"]:
                    publish(root, sampled, prior)
    print(json.dumps(data, indent=2))


def publish(root, work, data):
    dest = root / "results/flytrace"
    dest.mkdir(parents=True, exist_ok=True)
    rows, resources = [], []
    base = data["results"]["off"]["median_us"]
    for mode, result in data["results"].items():
        shutil.copyfile(work / mode / "final-isa.s", dest / f"{mode}.s")
        us = result["median_us"]
        rows.append(
            f"| [{mode}]({mode}.s) | {us:.3f} | {(us/base-1)*100:+.2f}% | 通过 | {result['trace']['records']} |"
        )
        r = result["isa"]
        resources.append(
            f"| {mode} | {r['vgpr']} | {r['agpr']} | {r['sgpr']} | {r['scratch']} | "
            f"{r['atomic_sites']} | {r['waterfalls']} | {r['mfma']} |"
        )
    shutil.copyfile(work / "tiles/timeline.json", dest / "timeline.json")
    all_text = ""
    all_path = root / "build-fly/flytrace/static/all/measurements.json"
    if all_path.exists():
        full = json.loads(all_path.read_text())
        if full["fingerprint"] == data["fingerprint"]:
            baseline = full["results"]["off"]["median_us"]
            all_rows = [
                f"| {mode} | {r['median_us']:.3f} | {(r['median_us']/baseline-1)*100:+.2f}% | "
                f"{r['trace']['records']} | {r['isa']['vgpr']} | {r['isa']['sgpr']} |"
                for mode, r in full["results"].items()
            ]
            all_text = (
                "\n全 grid 采集，1024 CTA / 4096 waves，全部数值、事件数和 tile 编号校验通过：\n\n"
                "| 版本 | GPU 耗时 µs | 相对 off | 记录数 | VGPR | SGPR |\n"
                "|---|---:|---:|---:|---:|---:|\n" + "\n".join(all_rows) + "\n"
            )
    body = (
        f"""# flytrace：静态布局、自动 capture

只看本文件；[timeline.json](timeline.json) 可直接用 Perfetto 打开。三份 `.s` 是本轮实际编译的 final ISA。

设备：{data['gpu']}；PyTorch：{data['torch']}。4096³ FP16，256 threads/CTA，32 KiB LDS。
以下采样 CTA (0,0,0) 的全部四个 wave；GPU event 计时始终覆盖整个 32×32 grid。

| 版本 / final ISA | GPU 耗时 µs | 相对 off | 数值校验 | 记录数 |
|---|---:|---:|---|---:|
"""
        + "\n".join(rows)
        + "\n"
        + all_text
        + """
| 单 CTA 采样版本 | VGPR | AGPR | SGPR | scratch B | atomic | waterfall | MFMA |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(resources)
        + f"""

每版本预热 20 次，{data['batches']} 批×{data['iterations']} 次 launch，轮换版本顺序，取批均值中位数。
输入准备、编译、自动 buffer 分配、同步回读和导出均不计入计时。
原始数据与全 grid ISA 放在 `build-fly/flytrace/static/`；前期方案对照放在 `build-fly/flytrace/optimization/`。

## 实现

- 编译器读取 `fly.trace_event`，为静态循环中的每次事件确定槽位。GPU 每条仅写完整低 32 位时间戳（4 B），事件名称、类型和可重建的循环 payload 保存在 host schema。
- 旧格式每记录三个 dword：时间戳低 32 位、event ID、int32 payload。现在槽位本身标识事件，循环索引由 host 重建，只保留一个时间 dword；记录写出量减少 2/3。每 wave 使用独立且按 64 B 对齐的区域。入口/出口各写一次完整 64-bit 时间，用于对齐和检测回绕窗口。
- 没有逐事件 atomic、动态游标、容量检查或 ID 打包。GEMM 计算循环直接写静态槽位，不修改 M0/EXEC，不维护寄存器页。全 grid 采集使用 LLVM 原生 `s_memrealtime` intrinsic，后端负责依赖等待；标量 `s_store_dword` 用一条 inline asm 表达。静态标记的核心为读时钟、等待、写出三条指令，循环地址计算可复用。
- 对只含标记和简单整数运算、每轮最多两个标记的顶层静态循环，编译器自动切换为 VGPR 缓存：每个标记位置用一个 VGPR 的 64 个 lane 保存 64 个时间戳，每满页合并写出。M0 在循环边界保存/恢复；普通长循环标记为 clock/wait/and/writelane/compare/branch 六条核心指令，页边界额外付出写出和等待成本。剩余页只写有效 lane；短循环在退出时写出。检测不满足条件时保留直写路径。
- 单 CTA 采样把 uniform guard 放在短 asm 中，避免它进入计算循环 CFG 并增加 VGPR 活跃范围。关闭 capture 时，标记消除，没有隐藏 trace 参数或采样指令。
- 仍有计时等待和 scalar-store 成本；它们会扰动流水。延迟写出实验让计时与 LDS 请求混在 LGKM 上，部分等待变成全等待，在该 GEMM 中更慢，因此没有采用。
- 相邻 tile 使用 boundary：一个时间戳同时结束上一 tile、开始下一 tile。零 gap 不表示插桩耗时为零；成本仍进入后续区间和外部 GPU 计时。

采用两个 lowering 路径是实测后的取舍。相同 70 条记录的 GEMM 中，强制寄存器缓存的单 CTA 开销为 +1.78%，直写为 +0.96%；全 grid 为 +3.57% 与 +3.68%，差异很小。因此 GEMM 保留直写。
简单密集循环的直写则会在下一次 `lgkmcnt(0)` 等待前次 scalar store，4096 次标记的间隔中位数约 280 ns；自动缓存避开了逐次显存等待。密集实验的普通点与满页点分开统计，见 `build-fly/flytrace/optimization/dense/`。
同一 4096 点微基准、四个 wave、五次采集共 81900 对间隔：缓存普通点中位数 80 ns，满页点 280 ns；包含满页点的平均间隔约 69.3 ns（全采集）/72.6 ns（带采样 guard）。这里是相邻读钟的实测间隔，包含循环指令，不是单条指令的独立延迟。
尝试用每个 lane 重复写一份时间戳、把写出量增加 64 倍，未优于静态标量写；增加内存不能解决计时等待和计算指令调度受到的扰动。

## 调用

```python
from flydsl.extension import flytrace

# @flyc.kernel 内：不需要 Plan、buffer 参数、begin/finish 或循环携带状态。
flytrace.boundary("load")
# ...
for k in range(K_TILES):
    flytrace.boundary("tile", k)
    # ... 原计算 ...
flytrace.end()

# @flyc.jit 内，在 kernel launch 前设置采样 CTA；None 表示全部 CTA。
with flytrace.configure(block=(0, 0, 0)):
    gemm_kernel(...).launch(...)

# host：原 launch 参数不变；分配、隐藏 ABI 参数和回读由 capture 管理。
with flytrace.capture("timeline.json"):
    launch_gemm(a, b, c)
```

支持 `mark`、`boundary/end` 和 `push/pop`；无需预先声明事件名。
`with flytrace.configure(block=...):` 只影响上下文内的 launch；支持嵌套，正常或异常退出时均恢复外层配置。不同配置分别生成 kernel specialization。配置为编译期常量，既不增加用户参数，也不生成运行时配置指令。
host 默认使用 JIT 的选择；没有 configure 时仍采样 CTA (0,0,0)。显式 `capture(block=...)` 优先，`capture(block=None)` 可临时覆盖为全 grid。
示例的 phases 版本排除 `k_tile`，tiles 版本排除 `mainloop/drain`，确保 64 个 tile 连续相接。
编译后的函数可在相同配置的新 capture 中复用；每个 capture 独立持有 buffer。
同一 capture 重复 launch 同一 specialization 时，保留最后一次记录。

## 当前限制与验证

仅 gfx942、完整收敛 wave64、静态 grid、1D block、常量边界且正步长的 `scf.for`。
包含 trace 的 runtime if/while 拒绝编译；payload 目前要求可由常量及循环索引重建且适合 int32。
仅默认 stream；capture 可跨多次顺序 launch，但不支持并发/嵌套 capture，也不累计每次 launch 的历史。
单 wave 采集跨度必须短于约 42.95 秒；保持全部 10 ns tick 精度，未用降低精度来压缩。

定向测试覆盖 CPU/CUDA 两种默认设备、4096 次记录、31/32/33/63/64/65 等页边界、
双标记交错写入与 M0 保存恢复、非单位步长嵌套循环、分配与行尾哨兵、
trace on/off 缓存分离、新 capture 的隐藏指针更新、artifact 序列化与不支持路径的拒绝。
trace、callstate 与 ATT 的 109 个相关用例在全局默认 CPU/CUDA 下分别通过。关闭 capture 的 final ISA 与原始 Example04 逐字节相同。
真实 Perfetto trace processor 导入验证：4 条 wave track，272 个 slice/instant，未结束区间 0；252 对相邻 tile 的 gap 全部为 0。
未运行全量测试或远端 CI。

```bash
PYTHONPATH=build-fly/python_packages:. /opt/venv/bin/python examples/04-flytrace_compare.py
PYTHONPATH=build-fly/python_packages:. /opt/venv/bin/python examples/04-flytrace_compare.py --all-blocks
PYTHONPATH=build-fly/python_packages:. /opt/venv/bin/python -m pytest -q tests/extension/test_flytrace.py
```
"""
    )
    (dest / "README.md").write_text(body)


if __name__ == "__main__":
    main()
