"""Execute an extracted TileRT specialization as one FlyDSL inline-ASM block."""

import hashlib
import re
import subprocess
from pathlib import Path

import yaml

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu
from flydsl.expr.typing import Int64, as_ir_value


def extract(code_object, symbol, llvm_bin, output):
    digest = hashlib.sha256(Path(code_object).read_bytes()).hexdigest()
    if digest != "6e0517e924f042d63a8b5a4e13f86696ae5bc4c1eee32765b28456d65bd2eb65":
        raise ValueError("The captured 560-byte ABI and segment PCs require the verified TileRT b14 code object")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    notes = subprocess.check_output([str(Path(llvm_bin) / "llvm-readelf"), "--notes", str(code_object)], text=True)
    metadata = yaml.safe_load(notes[notes.index("---") :])
    kernel = next(k for k in metadata["amdhsa.kernels"] if k[".name"] == symbol)
    if kernel[".kernarg_segment_size"] != 560 or kernel[".wavefront_size"] != 64:
        raise ValueError("Unexpected TileRT kernel ABI")
    dump = subprocess.check_output(
        [
            str(Path(llvm_bin) / "llvm-objdump"),
            "-d",
            "--mcpu=gfx950",
            "--disassemble-symbols=" + symbol,
            str(code_object),
        ],
        text=True,
    )
    instructions = []
    for line in dump.splitlines():
        match = re.match(r"\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f ]+)", line)
        if match:
            instructions.append((int(match[2], 16), match[1]))
    if not instructions:
        raise ValueError("No instructions found for " + symbol)
    addresses = {pc for pc, _ in instructions}
    lines = []
    for pc, inst in instructions:
        branch = re.fullmatch(r"(s_(?:cbranch_\w+|branch))\s+(-?\d+)", inst)
        if branch:
            displacement = int(branch[2])
            if displacement >= 32768:
                displacement -= 65536
            target = pc + 4 + displacement * 4
            if target not in addresses:
                raise ValueError(f"Branch leaves the extracted function: {pc:x} -> {target:x}")
            inst = f"{branch[1]} .Ltile_{target:x}"
        if "s_getpc" in inst or "s_swappc" in inst:
            raise ValueError("PC-relative data/calls need relocation before inlining")
        lines += [f".Ltile_{pc:x}:", inst]
    body = "\n".join(lines)
    (output / "original-disassembly.s").write_text(dump)
    (output / "inline-body.s").write_text(body)
    (output / "metadata.yaml").write_text(yaml.safe_dump(kernel))
    return body, kernel


def build(body, metadata, samples=1, replacement="none", trace=False):
    # The imported body owns the hardware register file until s_endpgm. Inputs
    # are pinned to the original ABI; all other referenced registers are clobbers.
    constraints = ["{s[0:1]}", "{s2}", "{v0}"]
    scalar_inputs, vector_inputs = set(), {0}
    vgprs = metadata[".vgpr_count"]
    if trace:
        if replacement != "none":
            raise ValueError("Trace and dispatch currently share reserved registers")
        from .trace import instrument

        body = instrument(body, vgprs, samples)
        vgprs += 4
        constraints += ["{s[90:91]}"]
        scalar_inputs.update((90, 91))
    if replacement == "dispatch":
        from .replacements import replace_dispatch

        body, vector_regs, _ = replace_dispatch(body, samples)
        scalar_inputs = {90, 91, 92}
        vector_inputs.update(vector_regs)
        constraints += ["{s90}", "{s91}", "{s92}"]
        constraints += [f"{{v{reg}}}" for reg in vector_regs]
    used_scalar = [int(n) for n in re.findall(r"\bs(\d+)\b", body)]
    used_scalar += [int(n) for n in re.findall(r"\bs\[\d+:(\d+)\]", body)]
    # s32 is reserved by LLVM as its scratch descriptor. The terminating ASM
    # never uses it; claiming it as a clobber is unnecessary and unsupported.
    constraints += [f"~{{s{i}}}" for i in range(3, max(used_scalar) + 1) if i not in scalar_inputs | {32}]
    constraints += [f"~{{v{i}}}" for i in range(1, vgprs) if i not in vector_inputs]
    constraints += ["~{vcc}", "~{scc}", "~{memory}"]
    constraint_string = ",".join(constraints)
    shared_bytes = metadata[".group_segment_fixed_size"]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def tile_inline(args: Int64, trace_buffer: Int64):
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        operands = [as_ir_value(args), as_ir_value(bid), as_ir_value(tid)]
        if fx.const_expr(trace):
            operands.append(as_ir_value(trace_buffer))
        if fx.const_expr(replacement == "dispatch"):
            group = bid >> 3
            rem = bid & 7
            shift = (bid < 128).select(fx.Int32(16), fx.Int32(-16))
            if fx.const_expr(samples == 1):
                task = (rem < 2).select(192 + (group + shift) * 2 + rem, group * 6 + rem - 2)
            elif fx.const_expr(samples == 2):
                task = (rem < 4).select(128 + (group + shift) * 4 + rem, group * 4 + rem - 4)
            else:
                task = (group + shift) * 8 + rem
            task = (task < 164 + samples).select(task, fx.Int32(-1))
            operands.extend(as_ir_value(x) for x in (group, rem, task, tid & 63, tid >> 6, (tid >> 4) & 3, tid & 15))
        llvm.InlineAsmOp(None, operands, body, constraint_string, has_side_effects=True)

    @flyc.jit
    def launch(args: Int64, trace_buffer: Int64 = 0, stream: fx.Stream = fx.Stream(None)):
        tile_inline(args, trace_buffer).launch(grid=(256,), block=(512,), smem=shared_bytes, stream=stream)

    return launch
