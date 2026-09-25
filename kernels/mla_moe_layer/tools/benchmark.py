# SPDX-License-Identifier: Apache-2.0
"""Graph latency of a fixed GLM-5 shard on 1/2/4/8 GPUs, with optional TileRT.

Both backends use the same generated weights and inputs. TileRT only supports
1 or 8 peers. These are fixed-shard tests (8 heads, intermediate size 256),
not full-model TP2/TP4 benchmarks. Each timed graph contains distinct epochs;
TileRT's captured scalar epochs require resetting mailboxes between replays.
"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import statistics
import struct
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernels.mla_moe_layer.op import Glm5MlaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.reference import KV_LORA, PE_DIM, make_weights, rope_table  # noqa: E402


def tilert_layer(weights, device):
    import tilert

    tilert.load_backend("glm5_2_rocm")
    from tilert.models.glm_5_2_rocm.ops import proj_wkvb, rmsnorm_projq_wqb, rmsnorm_projx_wqkva, unprojo_allreduce
    from tilert.models.glm_5_2_rocm.ops.pure_mla_moe_layer import PureMlaMoeLayerGlm5

    layer = PureMlaMoeLayerGlm5(device=device, num_heads=8, moe_w8a8=True)
    t, m = weights.t, layer.mla
    for mod, name, packer in (
        (m.m0, "qkv_a", rmsnorm_projx_wqkva.swizzle_weights_contig),
        (m.m1, "q_b", rmsnorm_projq_wqb.swizzle_weights_contig),
        (m.m4, "uk", proj_wkvb.swizzle_weights_contig),
        (m.m6, "uv", proj_wkvb.swizzle_weights_contig),
        (m.m7, "o", unprojo_allreduce.swizzle_v2),
    ):
        weight, scales = t["w_" + name], t["s_" + name]
        if name == "q_b":
            # FlyDSL groups [192 nope, 64 rope] within each head. TileRT
            # places every head's nope rows before every head's rope rows.
            blocks = torch.arange(8 * 4, device=device).reshape(8, 4)
            order64 = torch.cat((blocks[:, :3].flatten(), blocks[:, 3].flatten()))
            rows = (order64[:, None] * 64 + torch.arange(64, device=device)).flatten()
            weight = weight.view(torch.uint8)[rows].view(torch.float8_e4m3fn)
            scales = scales.repeat_interleave(2, dim=0)[order64]
        mod.packed = packer(weight.cpu()).to(device)
        mod.scales = scales
        if name in ("uk", "uv"):
            mod.scales = mod.scales.repeat_interleave(2, dim=0)
    m.m0.gamma_arg = t["g_in"].float()
    m.m1.gamma_arg = t["g_q"].float()
    m.m3.gamma = t["g_kv"].float()
    mo = layer.moe
    mo.front.router.w = t["w_r"].cpu()
    mo.front.router.gamma = t["g_post"].float()
    # TileRT stores the shared expert first; the FlyDSL reference stores it last.
    order = torch.tensor([256, *range(256)], device=device)
    mo.front.moe.w_fp8 = t["w_ug"].view(torch.uint8)[order].view(torch.float8_e4m3fn).cpu()
    mo.front.moe.scales = t["s_ug"][order]
    mo.down.w_fp8 = t["w_dn"].view(torch.uint8)[order].view(torch.float8_e4m3fn).cpu()
    mo.down.scales = t["s_dn"][order]
    layer.moe_banks()
    mo.front.moe.w_fp8 = mo.down.w_fp8 = None
    return layer


def worker(rank, args, port):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.npes)
    weights = make_weights(rank, heads=8, device=dev, seed=args.seed)
    layer = tilert_layer(weights, dev) if args.backend.startswith("tilert") else None
    cos, sin = rope_table(4096, device=dev)
    for samples in args.samples:
        gen = torch.Generator(device=dev).manual_seed(args.seed + 99)
        kv = torch.randn(4096, KV_LORA, generator=gen, device=dev).bfloat16()
        pe = torch.randn(4096, PE_DIM, generator=gen, device=dev).bfloat16()
        indices = torch.stack(
            [
                torch.randperm(max(args.pos + s + 1, 2048), generator=gen, device=dev)[:2048].sort().values
                for s in range(samples)
            ]
        ).int()
        if args.pos >= 2048:
            indices[:, -1] = torch.arange(args.pos, args.pos + samples, device=dev)
        h = torch.randn(samples, 6144, generator=gen, device=dev).bfloat16()
        x = torch.empty_like(h)
        pos = torch.tensor([args.pos], dtype=torch.int32, device=dev)
        if layer is None:
            op = Glm5MlaMoeLayer(weights, samples, rank=rank, npes=args.npes, timeline=args.trace)

            def run(epoch):
                op.forward(h, pos, kv, pe, indices, cos, sin, x_out=x, layer=epoch, advance=False)

            def advance():
                op.advance_step()

            def reset():
                pass

        else:
            from tilert.models.glm_5_2_rocm.ops import unprojo_allreduce as ua

            from kernels.comm.custom_all_reduce import FlyDSLAllreduce as Hip

            def peer_table():
                storage = torch.zeros(ua.sym_bytes(samples), dtype=torch.uint8, device=dev)
                base = Hip._get_alloc_base_ptr(storage.data_ptr())
                handles = [None] * args.npes
                dist.all_gather_object(handles, (Hip._get_mem_handle_bytes(base), storage.data_ptr() - base))
                addresses = [
                    storage.data_ptr() if r == rank else Hip._open_mem_handle(handle) + offset
                    for r, (handle, offset) in enumerate(handles)
                ]
                if args.npes in (2, 4):
                    addresses.extend([0] * (8 - args.npes))
                return torch.tensor(addresses, dtype=torch.int64, device=dev), storage

            sa, pa = peer_table()
            sf, pf = peer_table()
            parts = layer.alloc_partials(samples)
            uo = torch.empty_like(h)
            freqs = torch.stack(
                [cos[args.pos : args.pos + samples], sin[args.pos : args.pos + samples]], dim=-1
            ).reshape(samples, 64)
            ex = layer.mla.exchange(samples)
            reset_tensors = [
                ex.q_pairs,
                ex.kv_pairs,
                ex.pe_pairs,
                ex.m1_pairs,
                layer.mla.legs.qlines,
                layer.mla.legs.kvnew_pairs,
                layer.mla.legs.penew_pairs,
                layer.mla.tail.sen_a,
                layer.mla.tail.sen_b,
                layer.mla.tail.sen_proj,
                layer.moe.score_lines,
                layer.moe.flags,
                layer.moe.mid_pairs,
                layer.legs.hlines,
            ]
            native_npes = args.npes if args.npes in (1, 8) else 1
            native_rank = rank if args.npes in (1, 8) else 0

            def run(epoch):
                layer.forward(
                    h,
                    pos,
                    freqs,
                    pe[None],
                    kv[None],
                    indices,
                    parts,
                    weights.t["bias"],
                    residual=h,
                    sym_attn=sa,
                    sym_ffn=sf,
                    mype=native_rank,
                    npes=native_npes,
                    tag=epoch + 1,
                    sen_tag=epoch + 1,
                    ffn_flag=epoch + 1,
                    unproj_o=uo,
                    x_out=x,
                    reuse_selection=1,
                )

            def advance():
                pass

            def reset():
                for tensor in reset_tensors:
                    tensor.zero_()
                pa.zero_()
                pf.zero_()

        for i in range(5):
            run(i)
            advance()
        if args.backend == "tilert_inline":
            from asm_bridge.inline import build, extract

            hook = ctypes.CDLL(None)
            hook.capture_arm.argtypes = [ctypes.c_size_t]
            hook.capture_copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            hook.capture_copy.restype = ctypes.c_size_t
            hook.capture_name.restype = ctypes.c_char_p
            native_run = run
            captured = []
            for epoch in (0x123456, 0x234567):
                hook.capture_arm(560)
                native_run(epoch)
                torch.cuda.synchronize()
                raw = ctypes.create_string_buffer(560)
                size = hook.capture_copy(raw, 560)
                if size != 560:
                    raise RuntimeError(f"Captured {size} argument bytes, expected 560")
                captured.append(raw.raw)
            native_reference = x.clone()
            peer_patches = {}
            if args.npes in (2, 4):
                # Differential captures identify both attention and FFN copies
                # of rank/npes. The eight-peer launch is captured but skipped.
                hook.capture_arm_skip.argtypes = [ctypes.c_size_t]
                native_rank, native_npes = 7, 8
                hook.capture_arm_skip(560)
                native_run(0x123456)
                raw = ctypes.create_string_buffer(560)
                if hook.capture_copy(raw, 560) != 560:
                    raise RuntimeError("Failed to capture the peer ABI")
                for off in range(0, 560, 4):
                    before, after = (
                        struct.unpack_from("<I", captured[0], off)[0],
                        struct.unpack_from("<I", raw.raw, off)[0],
                    )
                    if before != after:
                        if (before, after) == (0, 7):
                            peer_patches[off] = rank
                        elif (before, after) == (1, 8):
                            peer_patches[off] = args.npes
                        else:
                            pointer_off = off // 8 * 8
                            before_ptr = struct.unpack_from("<Q", captured[0], pointer_off)[0]
                            after_ptr = struct.unpack_from("<Q", raw.raw, pointer_off)[0]
                            if not (before_ptr == 0 and after_ptr > 2**32):
                                raise RuntimeError(f"Unexpected peer ABI change at {off}: {before} -> {after}")
                if len(peer_patches) != 4:
                    raise RuntimeError(f"Expected two rank/npes pairs, found {peer_patches}")
                # NP8 initializes extra internal exchange pointers. Retain that
                # argument structure and patch only the verified rank/counts.
                captured[0] = raw.raw
                hook.capture_arm_skip(560)
                native_run(0x234567)
                if hook.capture_copy(raw, 560) != 560:
                    raise RuntimeError("Failed to capture the second peer epoch")
                captured[1] = raw.raw
            symbol = hook.capture_name().decode()
            print(f"Captured {symbol}", flush=True)
            offsets = [i for i in range(0, 560, 4) if captured[0][i : i + 4] != captured[1][i : i + 4]]
            if any(struct.unpack_from("<I", captured[0], i)[0] != 0x123457 for i in offsets):
                raise RuntimeError(f"Unexpected dynamic arguments at {offsets}")
            folder = Path(args.asm_artifacts) / f"s{samples}"
            # Each rank writes identical assembly to its own directory during extraction.
            body, metadata = extract(
                args.code_object, symbol, Path(os.environ["ROCM_PATH"]) / "llvm/bin", folder / f"rank{rank}"
            )
            (folder / f"rank{rank}" / "args.bin").write_bytes(captured[0])
            if peer_patches:
                from asm_bridge.peer_count import adapt

                body, metadata = adapt(body, metadata, samples, args.npes)
                print(f"[rank {rank}] adapted peer ABI: {peer_patches}", flush=True)
            if args.replace == "ffn":
                from asm_bridge.epilogue import build_ffn

                asm_launch = build_ffn(body, metadata, args.npes, samples)
            else:
                asm_launch = build(body, metadata, samples, args.replace, args.trace)
            if args.trace:
                from asm_bridge.trace import points

                POINTS = points(samples)
                trace_buffer = torch.zeros(256, len(POINTS), dtype=torch.int64, device=dev)
            else:
                trace_buffer = None
            blocks = []
            for epoch in range(args.layers):
                block = bytearray(captured[0])
                for offset in offsets:
                    struct.pack_into("<I", block, offset, epoch + 1)
                for offset, value in peer_patches.items():
                    struct.pack_into("<I", block, offset, value)
                block.extend(struct.pack("<QQII", x.data_ptr(), sf.data_ptr(), rank, epoch + 1))
                blocks.append(torch.tensor(list(block), dtype=torch.uint8, device=dev))
            if rank == 0:
                print(
                    f"INLINE {symbol}: tags={offsets}, VGPR={metadata['.vgpr_count']}, LDS={metadata['.group_segment_fixed_size']}",
                    flush=True,
                )

            def run(epoch):
                asm_launch(
                    blocks[epoch].data_ptr(),
                    0 if trace_buffer is None else trace_buffer.data_ptr(),
                    stream=torch.cuda.current_stream(),
                )

            reset()
            torch.cuda.synchronize()
            dist.barrier()
            run(0)
            torch.cuda.synchronize()
            if args.npes in (1, 8):
                torch.testing.assert_close(x, native_reference, atol=0, rtol=0)
                print(f"[rank {rank}] inline ASM equals native TileRT", flush=True)
            if args.npes in (2, 4) or args.verify_golden:
                from kernels.mla_moe_layer.reference import golden_layer

                def allreduce(value):
                    shards = [torch.empty_like(value.cpu()) for _ in range(args.npes)]
                    dist.all_gather(shards, value.cpu().contiguous())
                    return sum(shards[1:], shards[0]).to(dev)

                golden = golden_layer(weights, h, args.pos, kv.clone(), pe.clone(), indices, cos, sin, allreduce)
                expected = golden["x_out"]
                relative_l2 = ((x.float() - expected.float()).norm() / expected.float().norm()).item()
                attention = (
                    layer.legs.hlines.view(torch.bfloat16).view(4, 256, 8, 4)[:samples, :, :, :3].reshape(samples, 6144)
                )
                attn_l2 = ((attention.float() - golden["a"].float()).norm() / golden["a"].float().norm()).item()
                torch.save(
                    dict(x=x.cpu(), a=attention.cpu(), expected=expected.cpu(), expected_a=golden["a"].cpu()),
                    folder / f"rank{rank}" / "golden.pt",
                )
                print(f"[rank {rank}] golden attention L2={attn_l2:.6f}, output L2={relative_l2:.6f}", flush=True)
                if relative_l2 > 0.05 or not torch.isfinite(x).all():
                    raise AssertionError(f"Independent golden mismatch: relative L2={relative_l2}")
                print(f"[rank {rank}] independent golden relative L2={relative_l2:.6f}", flush=True)
            if args.changing_inputs:
                original_h = h.clone()
                if args.npes in (2, 4):
                    baseline_launch = build(body, metadata, samples)

                    def reference_run(epoch):
                        baseline_launch(blocks[0].data_ptr(), stream=torch.cuda.current_stream())

                else:
                    reference_run = native_run
                for iteration in range(args.changing_inputs):
                    h.copy_(torch.randn(h.shape, generator=gen, device=dev).bfloat16())
                    reset()
                    torch.cuda.synchronize()
                    dist.barrier()
                    reference_run(0x300000 + iteration)
                    torch.cuda.synchronize()
                    expected = x.clone()
                    reset()
                    torch.cuda.synchronize()
                    dist.barrier()
                    run(0)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(x, expected, atol=0, rtol=0)
                    ranks = [None] * args.npes
                    dist.all_gather_object(ranks, x.cpu())
                    for other in ranks:
                        torch.testing.assert_close(other, ranks[0], atol=0, rtol=0)
                print(f"[rank {rank}] {args.changing_inputs} changing inputs match baseline exactly", flush=True)
                h.copy_(original_h)
                reset()
                torch.cuda.synchronize()
                dist.barrier()
                run(0)
        torch.cuda.synchronize()
        reference = x.clone()
        if args.dump_outputs:
            Path(args.dump_outputs).mkdir(parents=True, exist_ok=True)
            torch.save(x.cpu(), Path(args.dump_outputs) / f"{args.backend}-tp{args.npes}-s{samples}-r{rank}.pt")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for epoch in range(args.layers):
                run(epoch)
            advance()
        times = []
        for repetition in range(args.repeats + 2):
            reset()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            torch.testing.assert_close(x, reference, atol=0, rtol=0)
            assert torch.isfinite(x).all()
            if repetition >= 2:
                times.append(start.elapsed_time(end) * 1000 / args.layers)
        per_rank = [None] * args.npes
        dist.all_gather_object(per_rank, times)
        if args.trace:
            if args.backend == "flydsl":
                folder = Path(args.asm_artifacts) / f"s{samples}" / f"rank{rank}"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(dict(stages=op.stages, ticks=op.timeline.cpu()), folder / "trace.pt")
                if rank == 0:
                    print(op.timeline_report(), flush=True)
            else:
                torch.save(dict(points=POINTS, ticks=trace_buffer.cpu()), folder / f"rank{rank}" / "trace.pt")
        if rank == 0:
            critical = [max(batch) for batch in zip(*per_rank)]
            result = dict(
                benchmark_version=2,
                backend=args.backend,
                replacement=args.replace,
                instrumented=args.trace,
                npes=args.npes,
                samples=samples,
                pos=args.pos,
                heads_per_rank=8,
                inter_per_rank=256,
                seed=args.seed,
                layers=args.layers,
                changing_inputs=args.changing_inputs,
                median_us=statistics.median(critical),
                min_us=min(critical),
                max_us=max(critical),
                rank_times_us=per_rank,
                kernel_sha256=hashlib.sha256(
                    (ROOT / "kernels/mla_moe_layer/glm5_mla_moe_layer.py").read_bytes()
                ).hexdigest(),
                torch=torch.__version__,
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bridge_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((Path(__file__).parent / "asm_bridge").glob("*.py"))
                },
            )
            print(json.dumps(result), flush=True)
            if args.output:
                with Path(args.output).open("a") as out:
                    out.write(json.dumps(result) + "\n")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flydsl", "tilert", "tilert_inline"), required=True)
    parser.add_argument("--npes", choices=(1, 2, 4, 8), type=int, required=True)
    parser.add_argument("--samples", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--pos", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output")
    parser.add_argument("--dump-outputs")
    parser.add_argument("--code-object", default="/root/tilert_isa/b14.co")
    parser.add_argument("--asm-artifacts", default="/root/glm5-perf-results/inline-asm")
    parser.add_argument("--replace", choices=("none", "dispatch", "ffn"), default="none")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--verify-golden", action="store_true")
    parser.add_argument("--changing-inputs", type=int, default=0)
    args = parser.parse_args()
    if args.backend == "tilert" and args.npes not in (1, 8):
        parser.error("TileRT's released whole-layer kernel only supports 1 or 8 peers")
    if args.trace and args.backend == "tilert":
        parser.error("--trace supports FlyDSL and inline TileRT")
    if args.replace != "none" and args.backend != "tilert_inline":
        parser.error("--replace requires --backend tilert_inline")
    if args.changing_inputs < 0 or (args.changing_inputs and args.backend != "tilert_inline"):
        parser.error("--changing-inputs requires a nonnegative count and the inline backend")
    if args.replace == "dispatch" and args.npes in (2, 4):
        parser.error("Dispatch and smaller-peer adaptation currently share reserved registers")
    if args.trace and args.backend == "tilert_inline" and (args.replace != "none" or args.npes in (2, 4)):
        parser.error("Inline tracing requires the intact 1/8-peer body")
    if not 1 <= args.layers <= 128:
        parser.error("layers must be in [1, 128] (the FlyDSL epoch stride)")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.npes)
