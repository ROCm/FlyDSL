"""Replace the S=1/2/4 TileRT FFN collective/residual/store with FlyDSL."""

import re

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64, T, as_ir_value
from kernels.common import buffer_ops as bo


def build_ffn(body, metadata, npes, samples=1):
    pc, np_reg, lds_base = {1: (0x196C0, 29, 7168), 2: (0x2CB74, 45, 6144), 4: (0x4170C, 25, 4096)}[samples]
    marker = f".Ltile_{pc:x}:"
    if marker not in body or f"s_cmp_lt_i32 s{np_reg}, 2" not in body:
        raise ValueError("Unrecognized FFN collective boundary")
    # The down pipeline has out-of-line basic blocks after the collective.
    # Preserve those branches and exit only after every down partial is in LDS.
    body = body.replace(marker, marker + "\ns_branch .Lfly_ffn")
    body += "\n.Lfly_ffn:\ns_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier\ns_mov_b64 exec, -1\n"
    scalar_max = max(
        [int(n) for n in re.findall(r"\bs(\d+)\b", body)] + [int(n) for n in re.findall(r"\bs\[\d+:(\d+)\]", body)]
    )
    # Unlike the terminating baseline, this ASM returns to compiler-generated
    # code. Declare the original ABI inputs as read/write so LLVM preserves
    # args, block ID and thread ID whenever they remain live after the ASM.
    constraints = ["={s[0:1]}", "={s2}", "={v0}", "0", "1", "2"]
    constraints += [f"~{{s{i}}}" for i in range(3, scalar_max + 1) if i != 32]
    constraints += [f"~{{v{i}}}" for i in range(1, metadata[".vgpr_count"])]
    constraints += ["~{vcc}", "~{scc}", "~{memory}"]
    constraint_string = ",".join(constraints)
    shared_bytes = metadata[".group_segment_fixed_size"]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def hybrid(args: Int64):
        bid, tid = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.thread_id("x"))
        llvm.InlineAsmOp(
            ir.Type.parse("!llvm.struct<(i64, i32, i32)>"),
            [as_ir_value(args), as_ir_value(bid), as_ir_value(tid)],
            body,
            constraint_string,
            has_side_effects=True,
        )

        def arg64(offset):
            ptr = fx.inttoptr(fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8), args + offset)
            return fx.Int64(fx.ptr_load(ptr))

        def shared_bf16(offset):
            ptr = fx.inttoptr(fx.PointerType.get(T.bf16, fx.AddressSpace.Shared, 2), fx.Int32(offset))
            return fx.Float32(fx.ptr_load(ptr))

        out = arg64(560)
        peers = arg64(568)
        rank_tag = arg64(576)
        rank = fx.Int32(rank_tag)
        tag = fx.Int32(rank_tag >> 32)
        lane, wave = tid & 63, tid >> 6
        sample = bid // (256 // samples)
        row = (bid % (256 // samples)) * (24 * samples) + lane * 2
        if const_expr(npes > 1):
            parity = (tag & 1) * (npes * samples * 6144 * 4)
            if wave < npes:
                peer_ptr = fx.inttoptr(fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8), peers + fx.Int64(wave) * 8)
                dst = fx.Int64(fx.ptr_load(peer_ptr))
                if lane < 12 * samples:
                    a = shared_bf16(lds_base + lane * 4)
                    b = shared_bf16(lds_base + 2 + lane * 4)
                    pair = fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    words = fx.Vector.from_elements([pair, tag], fx.Int32)
                    bo.buffer_store(
                        words,
                        bo.create_buffer_resource_from_addr(dst + fx.Int64(parity)),
                        (rank * samples + sample) * 6144 + row,
                        cache_modifier=17,
                    )
            gpu.barrier()
        if tid < 12 * samples:
            a = shared_bf16(lds_base + tid * 4)
            b = shared_bf16(lds_base + 2 + tid * 4)
            if const_expr(npes > 1):
                own_ptr = fx.inttoptr(fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8), peers + fx.Int64(rank) * 8)
                own = fx.Int64(fx.ptr_load(own_ptr)) + fx.Int64(parity)
                a, b = fx.Float32(0.0), fx.Float32(0.0)

                def read_peer(src):
                    ptr = fx.inttoptr(
                        fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8),
                        own + fx.Int64(((src * samples + sample) * 6144 + row) * 4),
                    )
                    return fx.Int64(fx.generic_load(ptr, memory_order=fx.AtomicOrdering.Monotonic, syncscope="one-as"))

                def pending(words):
                    bad = fx.Int32(words[0] >> 32) != tag
                    for i in range_constexpr(1, npes):
                        bad = bad | (fx.Int32(words[i] >> 32) != tag)
                    return bad

                words = [read_peer(src) for src in range(npes)]
                while pending(words):
                    words = [read_peer(src) for src in range(npes)]
                for src in range_constexpr(npes):
                    raw = fx.Int32(words[src])
                    a = a + (raw << 16).bitcast(fx.Float32)
                    b = b + (raw & fx.Int32(-65536)).bitcast(fx.Float32)
            a = a + shared_bf16(27152 + row * 2)
            b = b + shared_bf16(27154 + row * 2)
            values = fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16)
            bo.buffer_store(values, bo.create_buffer_resource_from_addr(out), sample * 6144 + row)

    @flyc.jit
    def launch(args: Int64, trace_buffer: Int64 = 0, stream: fx.Stream = fx.Stream(None)):
        hybrid(args).launch(grid=(256,), block=(512,), smem=shared_bytes, stream=stream)

    return launch
