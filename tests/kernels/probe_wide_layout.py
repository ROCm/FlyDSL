#!/usr/bin/env python3
"""Empirically decode the 32x32x64 f8f6f4 MFMA operand layout via the MMA identity.

C[m,n] = sum_k A[m,k] * B[n,k]   (M=N=32, K=64; acc 32x32 = v16f32 across 64 lanes).

Probe: set A to all-ones (A[m,k]=1 for all m,k). Then C[m,n] = sum_k B[n,k].
Set B so that lane L contributes a known per-(lane,byte) signature, and read C back to
learn which (n,k) each (lane, byte-in-i32x8) of the B operand feeds. Comparing against the
narrow op's known mapping tells us how to stage V for the wide op.

We dump the full acc for a chosen B pattern; the host decodes. Single wave.
"""

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw

WARP = 64


def build():
    @flyc.kernel
    def probe(Abytes: fx.Tensor, Bbytes: fx.Tensor, OUT: fx.Tensor):
        # A/Bbytes: [64, 32] int8 operands (32 fp8/lane). OUT: [64, 16] f32 acc per lane.
        v16f32 = Vec.make_type(16, fx.Float32)
        lane = fx.Index(gpu.thread_idx.x) % fx.Index(WARP)
        Adiv = fx.logical_divide(fx.rocdl.make_buffer_tensor(Abytes), fx.make_layout(1, 1))
        Bdiv = fx.logical_divide(fx.rocdl.make_buffer_tensor(Bbytes), fx.make_layout(1, 1))
        Odiv = fx.logical_divide(fx.rocdl.make_buffer_tensor(OUT), fx.make_layout(1, 1))
        load_b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        v4i32 = Vec.make_type(4, fx.Int32)

        # Load this lane's 32 B bytes (= 8 i32) as two v4i32 (lane*32 .. +16 bytes).
        base = lane * fx.Index(32)
        b_lo = fly.copy_atom_call_ssa([v4i32], load_b, fx.slice(Bdiv, (None, fx.Int32(base))))
        b_hi = fly.copy_atom_call_ssa([v4i32], load_b, fx.slice(Bdiv, (None, fx.Int32(base + fx.Index(16)))))
        b_op = Vec(b_lo).shuffle(Vec(b_hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        # A operand: read from a second input so the host can select which K-bytes are 1.0.
        a_lo = fly.copy_atom_call_ssa([v4i32], load_b, fx.slice(Adiv, (None, fx.Int32(base))))
        a_hi = fly.copy_atom_call_ssa([v4i32], load_b, fx.slice(Adiv, (None, fx.Int32(base + fx.Index(16)))))
        a_op = Vec(a_lo).shuffle(Vec(a_hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        c0 = Vec.from_elements([fx.Float32(0.0) for _ in range_constexpr(16)], fx.Float32).ir_value()
        acc = rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            v16f32,
            _raw(a_op),
            _raw(b_op),
            _raw(c0),
            0,
            0,
            0,
            _raw(fx.Int32(0x7F7F7F7F)),
            0,
            _raw(fx.Int32(0x7F7F7F7F)),
        ).result
        accv = Vec(acc, (16,), fx.Float32)
        oregf = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        storef = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        for r in range_constexpr(16):
            pack = Vec.from_elements([fx.Float32(accv[r])], fx.Float32)
            fx.memref_store_vec(pack, oregf)
            fx.copy(storef, oregf, fx.slice(Odiv, (None, fx.Int32(lane * fx.Index(16) + fx.Index(r)))))

    return probe


def build_launch(kern):
    @flyc.jit
    def launch(Abytes: fx.Tensor, Bbytes: fx.Tensor, OUT: fx.Tensor):
        kern(Abytes, Bbytes, OUT).launch(grid=(1, 1, 1), block=(WARP, 1, 1))

    return launch


def build_permlane_probe():
    """On-device check of the permlane32_swap(a, a) lane-half convention.

    The wide-P in-register gather (_read_p_fp8_wide_shuffle) depends on the fact that
    permlane32_swap(a, a) returns this lane's +/-32 partner value in result[1] for low-half
    lanes (lane//32 == 0) but result[0] for high-half lanes (lane//32 == 1) -- the same
    convention the proven O-store _swap_halves uses. Reading result[1] unconditionally (the
    old bug) silently fed high-half lanes their OWN value. This kernel fills each lane's dword
    with its lane id, applies the FIXED lane-half select, and writes the result so the host
    can assert out[L] == (L ^ 32) for every lane.
    """

    @flyc.kernel
    def probe(OUT: fx.Tensor):
        lane = fx.Index(gpu.thread_idx.x) % fx.Index(WARP)
        pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        # Each lane's source value = its own lane id.
        x = fx.Int32(gpu.thread_idx.x) % fx.Int32(WARP)
        sw = rocdl.permlane32_swap(pair_ty, _raw(x), _raw(x), False, False)
        lo_res = llvm.extractvalue(T.i32, sw, [0])
        hi_res = llvm.extractvalue(T.i32, sw, [1])
        is_hi = ArithValue(fx.Int32(lane // fx.Index(32)) == fx.Int32(1))
        partner = fx.Int32(is_hi.select(lo_res, hi_res))
        Odiv = fx.logical_divide(fx.rocdl.make_buffer_tensor(OUT), fx.make_layout(1, 1))
        store32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        oreg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        fx.memref_store_vec(Vec.from_elements([partner], fx.Int32), oreg)
        fx.copy(store32, oreg, fx.slice(Odiv, (None, fx.Int32(lane))))

    @flyc.jit
    def launch(OUT: fx.Tensor):
        probe(OUT).launch(grid=(1, 1, 1), block=(WARP, 1, 1))

    return launch


# ---- Analytical per-dword layout model (mirrors kernels/flash_attn_gfx950.py) ----
# Single wave: wave_id == 0, so q_local == lane % 32. These formulas are copied verbatim
# from _stage_p_fp8_wide / _read_p_fp8_wide / _read_p_fp8_wide_shuffle and PV_K_STEPS == 2.
PV_K_STEPS = 2


def narrow_p_source_byte(lane, pks, s):
    """The (query_row, key) that narrow-P pack byte (pks, s) of `lane` carries.

    From _stage_p_fp8_wide: q_local = lane % 32; half = (lane // 32) * 4; r = pks*8 + s;
    the lo byte -> key = half + (r//4)*8 + (r%4); the hi byte -> that key + 32.
    """
    q = lane % 32
    half = (lane // 32) * 4
    r = pks * 8 + s
    key_lo = half + (r // 4) * 8 + (r % 4)
    return (q, key_lo), (q, key_lo + 32)  # (lo-strip (q,key), hi-strip (q,key))


def lds_wide_dest(lane):
    """LDS-gather wide operand for dest `lane`: 32 bytes b -> (q=lane%32, key=(lane//32)*32+b).

    From _read_p_fp8_wide: row = q*_PF_ROW + (lane//32)*32; byte b reads key (lane//32)*32 + b.
    Returns list of 8 dwords, each a tuple of 4 (q, key).
    """
    q = lane % 32
    base_key = (lane // 32) * 32
    bytes_ = [(q, base_key + b) for b in range(32)]
    return [tuple(bytes_[d * 4 : d * 4 + 4]) for d in range(8)]


def shuffle_wide_dest(lane):
    """Shuffle-gather wide operand for dest `lane`, modeling _read_p_fp8_wide_shuffle.

    For each pks in 0..1 and dword d in 0..1: own strip value is this lane's lo (h=0) / hi
    (h=1) dword d; partner is lane^32's lo/hi dword d (the FIXED lane-half permlane select).
    even dword := h ? partner : own; odd := h ? own : partner. Output dwords appended
    (even, odd) per (pks, d) -> 8 dwords. Each dword carries 4 bytes (q, key) from a narrow
    pack's lo/hi 32-bit word. lo word of pack pks holds bytes s=0..3 (-> dword d=0) and s=4..7
    (-> d=1); hi word the +32-key variants.
    """
    h = lane // 32
    partner_lane = lane ^ 32

    def strip_dword(src_lane, pks, d, hi):
        # The narrow lo/hi 32-bit word for pack pks, dword d (s = d*4 .. d*4+3), hi or lo strip.
        out = []
        for s in range(d * 4, d * 4 + 4):
            lo_qk, hi_qk = narrow_p_source_byte(src_lane, pks, s)
            out.append(hi_qk if hi else lo_qk)
        return tuple(out)

    words = []
    for pks in range(PV_K_STEPS):
        for d in range(2):
            own = strip_dword(lane, pks, d, hi=(h == 1))
            partner = strip_dword(partner_lane, pks, d, hi=(h == 1))
            even = partner if h == 1 else own
            odd = own if h == 1 else partner
            words.append(even)
            words.append(odd)
    return words


def analytical_dword_diff():
    """Compare LDS-gather vs shuffle-gather wide-P layouts for all lanes/dwords.

    Returns (n_mismatch, report_lines). A mismatch means the two gather paths would deliver a
    different (query, key) for some destination dword -- i.e. the operands would disagree.
    """
    n_mismatch = 0
    lines = []
    for lane in range(64):
        lds = lds_wide_dest(lane)
        shuf = shuffle_wide_dest(lane)
        for d in range(8):
            if lds[d] != shuf[d]:
                n_mismatch += 1
                lines.append(f"  MISMATCH lane{lane:02d} dword{d}: lds={lds[d]} shuf={shuf[d]}")
    return n_mismatch, lines


def main():
    dev = "cuda"
    exe = build_launch(build())
    ones = torch.full((64, 32), 0x38, dtype=torch.uint8, device=dev)  # all 1.0

    # Sanity.
    OUT = torch.zeros((64, 16), dtype=torch.float32, device=dev)
    exe(ones.view(-1), ones.view(-1), OUT.view(-1))
    torch.cuda.synchronize()
    print("sanity all-ones acc unique:", np.unique(OUT.cpu().numpy()), "(expect [64])")

    # Decode the A operand's (lane,byte) -> (m, k):
    #   B = all-ones, A has ONE byte = 1.0 at (lane L, byte p). Then
    #   C[m,n] = sum_k A[m,k]*1 = 1 for the single m that (L,p) feeds (all n).
    #   The lit output (out_lane, r) gives m; varying p at fixed L shows how bytes map to
    #   distinct m's (=> the K contraction index is encoded by which bytes share an m).
    # acc element (out_lane, r): for 32x32 v16f32, row m = ? We just record the raw hit.
    # K-decode: A single byte (0,pa)=1 => one (m=row(0,pa), k=K(0,pa)).
    #           B single byte (lb,pb)=1 => one (n, k=K(lb,pb)).
    # C nonzero iff the two k's match. Fix A at lane0 and scan its byte pa; for each, find
    # which (B lane, B byte) lights C -> reveals K(A:0,pa) == K(B:lb,pb).
    # This tells us, for the B(=P) operand, which (lane,byte) holds which contraction k.
    print("\n=== K-match: A(lane0,pa)=1 vs B(laneLb,pb)=1 -> which B(lane,byte) shares k ===")
    for pa in [0, 1, 2, 3, 4, 8, 16, 31]:
        A = torch.zeros((64, 32), dtype=torch.uint8, device=dev)
        A[0, pa] = 0x38
        matches = []
        for Lb in [0, 32]:
            for pb in range(32):
                B = torch.zeros((64, 32), dtype=torch.uint8, device=dev)
                B[Lb, pb] = 0x38
                OUT = torch.zeros((64, 16), dtype=torch.float32, device=dev)
                exe(A.view(-1), B.view(-1), OUT.view(-1))
                torch.cuda.synchronize()
                if np.abs(OUT.cpu().numpy()).max() > 1e-3:
                    matches.append(f"B(L{Lb},p{pb})")
        print(f"  A(L0,p{pa:02d}) k-matches: {matches}")

    # --- shuffle-vs-LDS per-dword wide-P layout diff probe ---
    # 1) On-device check of the permlane32 lane-half convention (the actual route-A bug).
    print("\n=== permlane32 lane-half convention (expect out[L] == L^32 for all lanes) ===")
    perm_exe = build_permlane_probe()
    PO = torch.full((64,), -1, dtype=torch.int32, device=dev)
    perm_exe(PO.view(-1))
    torch.cuda.synchronize()
    po = PO.cpu().numpy()
    expect = np.array([L ^ 32 for L in range(64)], dtype=np.int32)
    perm_bad = int((po != expect).sum())
    if perm_bad:
        for L in range(64):
            if po[L] != expect[L]:
                print(f"  PERMLANE MISMATCH lane{L:02d}: got {po[L]}, expect {expect[L]}")
    print(f"permlane32 lane-half: {'PASS' if perm_bad == 0 else f'FAIL ({perm_bad} lanes)'}")

    # 2) Analytical per-dword diff of the two wide-P gather paths (LDS vs in-register shuffle).
    #    With the FIXED shuffle, every destination dword must carry the same (query, key) as the
    #    LDS gather; any difference is a layout bug.
    print("\n=== shuffle-vs-LDS wide-P per-dword diff (expect 0 mismatches) ===")
    n_mismatch, lines = analytical_dword_diff()
    for ln in lines[:32]:
        print(ln)
    print(f"per-dword diff: {'PASS (0 mismatches)' if n_mismatch == 0 else f'FAIL ({n_mismatch} dwords)'}")

    ok = perm_bad == 0 and n_mismatch == 0
    print(f"\nPROBE RESULT: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
