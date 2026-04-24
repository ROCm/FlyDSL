# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TopK Gating Softmax kernel builder using the @flyc.kernel API.

Fuses softmax + top-K selection + optional renormalization for MoE gating:

  1. softmax(logits)  = exp(x - max(x)) / sum(exp(x - max(x)))
  2. top-K selection   = K iterations of argmax-then-mask
  3. renormalize       = rescale K selected weights to sum to 1.0

Outputs: topk_weights (f32), topk_indices (i32), token_expert_indices (i32).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32

from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size


KERNEL_NAME = "topk_gating_softmax_kernel"

WARP_SIZE = get_warp_size()
WARPS_PER_BLOCK = 4
BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE  # 256 on gfx95x


def _pick_layout(num_experts: int):
    """Pick (VPT, THREADS_PER_TOKEN) for the multi-token-per-block fast path.

    Constraints:
      - ``VPT`` is a power of 2 in [1, 16]
      - ``THREADS_PER_TOKEN = num_experts // VPT`` is a power of 2 <= WARP_SIZE
      - prefer the largest ``VPT`` (fewest loads, widest atom)

    For ``num_experts=128`` on a 64-wide wave this picks ``(VPT=16, TPT=8)``
    (TOKENS_PER_BLOCK=32). vLLM's ``topkGatingSoftmax`` uses VPT=8 / TPT=16
    """
    for vpt in [16, 8, 4, 2, 1]:
        if num_experts % vpt != 0:
            continue
        tpt = num_experts // vpt
        if tpt > WARP_SIZE:
            continue
        if (tpt & (tpt - 1)) != 0:
            continue
        return vpt, tpt
    return None, None


def build_topk_gating_softmax_module(
    num_experts: int,
    topk: int,
    dtype_str: str = "bf16",
    renormalize: bool = True,
):
    """Build a fused TopK gating softmax kernel.

    Args:
        num_experts: Number of MoE experts (columns in gating_output).
        topk: Number of top experts to select per token.
        dtype_str: Input data type ('f32', 'f16', 'bf16').
        renormalize: If True, rescale selected weights to sum to 1.

    Returns:
        A @flyc.jit launcher function with signature
        ``(gating, weights, indices, tei, num_tokens, *, stream)``.
    """
    arch = get_hip_arch()

    elem_bits = 32 if dtype_str == "f32" else 16

    VPT, THREADS_PER_TOKEN = _pick_layout(num_experts)
    if VPT is None:
        raise ValueError(
            f"num_experts={num_experts} is not supported by the multi-token-per-block "
            f"layout: requires num_experts // VPT to be a power of 2 <= "
            f"WARP_SIZE={WARP_SIZE} for some VPT in [16, 8, 4, 2, 1]."
        )

    TOKENS_PER_WARP = WARP_SIZE // THREADS_PER_TOKEN
    TOKENS_PER_BLOCK = WARPS_PER_BLOCK * TOKENS_PER_WARP

    if topk > num_experts:
        raise ValueError(f"topk={topk} > num_experts={num_experts}")

    if elem_bits <= 16 and VPT % 8 == 0:
        ATOM_BITS = 128                     # 8 bf16/f16 per atom call
    elif elem_bits <= 16 and VPT % 4 == 0:
        ATOM_BITS = 64                      # 4 bf16/f16 per atom call
    elif elem_bits <= 16 and VPT % 2 == 0:
        ATOM_BITS = 32                      # 2 bf16/f16 per atom call
    elif elem_bits == 32 and VPT % 2 == 0:
        ATOM_BITS = 64                      # 2 f32 per atom call
    else:
        ATOM_BITS = elem_bits               # 1 element per atom call
    ELEMS_PER_ATOM = ATOM_BITS // elem_bits
    ATOMS_PER_THREAD = VPT // ELEMS_PER_ATOM

    # No shared memory used — every reduction stays inside a sub-warp lane group.
    allocator = SmemAllocator(None, arch=arch)

    @flyc.kernel
    def topk_gating_softmax_kernel(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIndices: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
        i32_num_tokens: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast

        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_log2e = arith.constant(1.4426950408889634, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)

        # ── Thread → (warp, token-in-warp, expert-lane) decomposition ────
        c_warp = fx.Int32(WARP_SIZE)
        c_tpt = fx.Int32(THREADS_PER_TOKEN)
        c_tpw = fx.Int32(TOKENS_PER_WARP)
        c_tpb = fx.Int32(TOKENS_PER_BLOCK)
        c_vpt = fx.Int32(VPT)

        warp_id = tid // c_warp                          # 0..WARPS_PER_BLOCK-1
        lane = tid % c_warp                              # 0..WARP_SIZE-1
        token_in_warp = lane // c_tpt                    # 0..TOKENS_PER_WARP-1
        expert_lane = lane % c_tpt                       # 0..THREADS_PER_TOKEN-1
        local_token = warp_id * c_tpw + token_in_warp    # 0..TOKENS_PER_BLOCK-1
        global_token = bid * c_tpb + local_token         # token row

        in_range = global_token < i32_num_tokens

        global_token_safe = in_range.select(global_token, fx.Int32(0))

        # ── Sub-warp reductions over the THREADS_PER_TOKEN-lane group ────
        def group_reduce(x, mode):
            """Butterfly reduce within a THREADS_PER_TOKEN sub-warp group."""
            width_i32 = c_tpt
            w = x
            for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
                off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
                peer = w.shuffle_xor(off, width_i32)
                if mode == "max":
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def group_reduce_argmax(val, idx):
            """Butterfly argmax within a THREADS_PER_TOKEN sub-warp group.

            All lanes in the group end with the same (max_val, max_idx).
            Ties are broken by the lower expert index.
            """
            width_i32 = c_tpt
            wv, wi = val, idx
            for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
                off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
                peer_v = wv.shuffle_xor(off, width_i32)
                peer_i = wi.shuffle_xor(off, width_i32)
                is_greater = peer_v > wv
                is_equal = ArithValue(peer_v) == ArithValue(wv)
                peer_lower_idx = peer_i < wi
                take_peer = is_greater | (is_equal & peer_lower_idx)
                wv = take_peer.select(peer_v, wv)
                wi = take_peer.select(peer_i, wi)
            return wv, wi

        # ── Buffer-backed views ──────────────────────────────────────────
        GatingOutput_buf = fx.rocdl.make_buffer_tensor(GatingOutput)
        TopkWeights_buf = fx.rocdl.make_buffer_tensor(TopkWeights)
        TopkIndices_buf = fx.rocdl.make_buffer_tensor(TopkIndices)
        TokenExpertIndices_buf = fx.rocdl.make_buffer_tensor(TokenExpertIndices)

        # Per-thread row slices (different threads serve different tokens).
        row_gating = fx.slice(GatingOutput_buf, (global_token_safe, None))
        row_weights = fx.slice(TopkWeights_buf, (global_token_safe, None))
        row_indices = fx.slice(TopkIndices_buf, (global_token_safe, None))
        row_tei = fx.slice(TokenExpertIndices_buf, (global_token_safe, None))

        # Per-element scalar tiling for the K-wide output rows. The gating
        # row is divided into ELEMS_PER_ATOM-wide chunks for input loads.
        gating_div = fx.logical_divide(
            row_gating, fx.make_layout(ELEMS_PER_ATOM, 1)
        )
        weights_div = fx.logical_divide(row_weights, fx.make_layout(1, 1))
        indices_div = fx.logical_divide(row_indices, fx.make_layout(1, 1))
        tei_div = fx.logical_divide(row_tei, fx.make_layout(1, 1))

        # ── Input load: ATOM_BITS-wide buffer copy (ELEMS_PER_ATOM elems) ─
        copy_atom_in = fx.make_copy_atom(fx.rocdl.BufferCopy(ATOM_BITS), elem_bits)
        atom_reg_ty_in = fx.MemRefType.get(
            elem_type,
            fx.LayoutType.get(ELEMS_PER_ATOM, 1),
            fx.AddressSpace.Register,
        )
        atom_reg_lay_in = fx.make_layout(ELEMS_PER_ATOM, 1)

        # Output copy atoms: f32 path is reused for i32 indices via bitcast
        # (callers pass torch.float32 views over int32 storage; see comment
        # near `_store_scalar_i32` below).
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        scalar_reg_ty_f32 = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        scalar_reg_lay = fx.make_layout(1, 1)

        def _load_atom_in(divided, atom_index):
            """Load ELEMS_PER_ATOM contiguous elements starting at atom_index."""
            view = fx.slice(divided, (None, atom_index))
            r = fx.memref_alloca(atom_reg_ty_in, atom_reg_lay_in)
            fx.copy_atom_call(copy_atom_in, view, r)
            return fx.memref_load_vec(r)

        def _store_scalar_f32(divided, index, val):
            r = fx.memref_alloca(scalar_reg_ty_f32, scalar_reg_lay)
            vec_ty = T.vec(1, T.f32)
            v = vector.from_elements(vec_ty, [val])
            fx.memref_store_vec(v, r)
            view = fx.slice(divided, (None, index))
            fx.copy_atom_call(copy_atom_f32, r, view)

        def _store_scalar_i32(divided, index, val):
            # `divided` is a logical_divide of a torch.float32-viewed buffer,
            # so its element type is f32. Reinterpret the i32 bits as f32 and
            # store via the f32 copy atom (avoids signed-vs-signless legalize
            # failures when going through si32).
            val_f32 = ArithValue(val).bitcast(T.f32)
            r = fx.memref_alloca(scalar_reg_ty_f32, scalar_reg_lay)
            vec_ty = T.vec(1, T.f32)
            v = vector.from_elements(vec_ty, [val_f32])
            fx.memref_store_vec(v, r)
            view = fx.slice(divided, (None, index))
            fx.copy_atom_call(copy_atom_f32, r, view)

        # ==================================================================
        # Pass 1: Load this thread's VPT experts + per-thread max
        # ==================================================================
        # Each thread owns the contiguous expert columns
        # [expert_lane * VPT, expert_lane * VPT + VPT). With THREADS_PER_TOKEN
        # = num_experts / VPT, every column in [0, num_experts) is covered
        # exactly once across the THREADS_PER_TOKEN-lane group.
        # We issue ATOMS_PER_THREAD wide loads (each ELEMS_PER_ATOM elements),
        # then unpack into a flat per-element list.
        col_idx_list = []
        for v in range_constexpr(VPT):
            col_idx_list.append(expert_lane * c_vpt + fx.Int32(v))

        c_atoms_pt = fx.Int32(ATOMS_PER_THREAD)
        x_list = []
        thread_max = c_neg_inf
        for a in range_constexpr(ATOMS_PER_THREAD):
            atom_idx = expert_lane * c_atoms_pt + fx.Int32(a)
            atom_vec = _load_atom_in(gating_div, atom_idx)
            for v in range_constexpr(ELEMS_PER_ATOM):
                val_e = vector.extract(atom_vec, static_position=[v])
                xv = val_e if dtype_str == "f32" else val_e.extf(compute_type)
                x_list.append(xv)
                thread_max = thread_max.maximumf(xv)

        group_max = group_reduce(thread_max, "max")

        # ==================================================================
        # Pass 2: exp(x - max) and per-token sum
        # ==================================================================
        thread_sum = c_zero_f
        exp_list = []
        for v in range_constexpr(VPT):
            sub = ArithValue(x_list[v]) - ArithValue(group_max)
            scaled = sub * c_log2e
            ev = scaled.exp2(fastmath=fm_fast)
            exp_list.append(ev)
            thread_sum = thread_sum + ev

        group_sum = group_reduce(thread_sum, "sum")

        # ==================================================================
        # Pass 3: Normalize -> softmax probabilities (kept in registers)
        # ==================================================================
        inv_sum = c_one_f / ArithValue(group_sum)
        prob_list = []
        for v in range_constexpr(VPT):
            prob_list.append(ArithValue(exp_list[v]) * ArithValue(inv_sum))

        # ==================================================================
        # Pass 4: Iterative Top-K (sub-warp argmax → mask)
        # ==================================================================
        # Stash both the winning weight and index per iteration so Pass 5
        # can write them without recomputing.
        selected_weights = []   # one f32 per k iter (replicated across the group)
        selected_indices = []   # one i32 per k iter (replicated across the group)
        selected_sum = c_zero_f

        for k_idx in range_constexpr(topk):
            # Per-thread argmax over its VPT slots.
            thread_best_val = c_neg_inf
            thread_best_idx = fx.Int32(-1)
            for v in range_constexpr(VPT):
                pv = prob_list[v]
                ci = col_idx_list[v]
                is_better = pv > thread_best_val
                thread_best_val = is_better.select(pv, thread_best_val)
                thread_best_idx = is_better.select(ci, thread_best_idx)

            # Sub-warp argmax → all THREADS_PER_TOKEN lanes hold the winner.
            global_best_val, global_best_idx = group_reduce_argmax(
                thread_best_val, thread_best_idx
            )

            selected_weights.append(global_best_val)
            selected_indices.append(global_best_idx)
            selected_sum = selected_sum + global_best_val

            # Mask the winner out of every thread's local prob slots so
            # the next iteration finds the runner-up.
            for v in range_constexpr(VPT):
                ci = col_idx_list[v]
                is_winner = ArithValue(ci) == ArithValue(global_best_idx)
                prob_list[v] = is_winner.select(c_neg_inf, prob_list[v])

        # ==================================================================
        # Pass 5: Leader writes weights/indices/tei (with optional renorm)
        # ==================================================================
        # Only the leader of each in-range token group touches global memory.
        if renormalize:
            c_eps = arith.constant(1e-20, type=compute_type)
            denom = selected_sum.maximumf(c_eps)
            inv_denom = c_one_f / ArithValue(denom)

        # Inline the leader-active predicate so the AST rewriter recognises it
        # as a dynamic test (it must contain a Call) and lowers `if ...` to
        # `scf.IfOp`. Wrapping it in a named variable would short-circuit the
        # rewrite and the runtime would try `Boolean.__bool__()` and raise.
        if (expert_lane == fx.Int32(0)) & (global_token < i32_num_tokens):
            num_tokens_v = ArithValue(i32_num_tokens)
            for k_idx in range_constexpr(topk):
                w_val = selected_weights[k_idx]
                if renormalize:
                    w_val = ArithValue(w_val) * ArithValue(inv_denom)
                _store_scalar_f32(weights_div, Int32(k_idx), w_val)
                _store_scalar_i32(indices_div, Int32(k_idx), selected_indices[k_idx])
                # tei[t, k] = k * num_tokens + t  (matches vLLM convention)
                tei_val = Int32(k_idx) * num_tokens_v + global_token
                _store_scalar_i32(tei_div, Int32(k_idx), tei_val)

    # ── JIT host launcher ─────────────────────────────────────────────────
    @flyc.jit
    def launch_topk_gating_softmax(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIndices: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
        num_tokens_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # grid_x = ceil(num_tokens / TOKENS_PER_BLOCK).
        # We use the (n - 1) // tpb + 1 form (valid for n >= 1) since the
        # additive (n + tpb - 1) form was producing the wrong grid count
        # under JIT specialization in this DSL.
        c_tpb_idx = fx.Index(TOKENS_PER_BLOCK)
        c_one_idx = fx.Index(1)
        nt_idx = arith.index_cast(T.index, num_tokens_in)
        grid_x = (nt_idx - c_one_idx) // c_tpb_idx + c_one_idx

        launcher = topk_gating_softmax_kernel(
            GatingOutput,
            TopkWeights,
            TopkIndices,
            TokenExpertIndices,
            num_tokens_in,
        )
        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_gating_softmax
