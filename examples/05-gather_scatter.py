# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Row gather/scatter with explicit offset tensors.

Gather computes ``Dst[r, :] = Src[Idx[r], :]`` and scatter computes
``Dst[Idx[r], :] = Src[r, :]``. The example keeps the indirection in ordinary
layout algebra: ``Idx`` is broadcast to a per-copy offset tensor, then
``fx.gather`` / ``fx.scatter`` apply those offsets around a standard
``TiledCopy`` fragment.

The data path uses ``UniversalCopy128b`` for four contiguous ``Float32`` values
per copy instance. ``Idx`` remains a separate ``UniversalCopy32b`` path with one
``Int32`` index per instance.

Argument layout contract for ``fx.gather`` / ``fx.scatter``:

.. code-block:: text

    data fragment : ((AtomV, TV), Rest...)
    offset tensor : (TV, Rest...)
    pred tensor   : (TV, Rest...)  optional

"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

TileM = 8
TileN = 32


@flyc.kernel
def gather_kernel(
    Src: fx.Tensor,  # (M, N) f32
    Dst: fx.Tensor,  # (R, N) f32
    Idx: fx.Tensor,  # (R,) i32 row indices
):
    tid = fx.thread_idx.x
    bid_m, bid_n, _ = fx.block_idx

    atom_f32 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

    thr_copy = fx.make_tiled_copy_tv(
        atom_f32,
        fx.make_layout((TileM, TileN // 4), (TileN // 4, 1)),
        fx.make_layout((1, 4), (1, 1)),
    ).get_slice(tid)

    R, N = Dst.shape.unpack()

    # Broadcast index tensor to a per-copy offset tensor.
    idx_broadcast = fx.make_view(fx.get_iter(Idx), fx.make_layout((R, N), (1, 0)))
    # Coordinate tensors for column indices inside the current tile.
    Col = fx.make_view(fx.make_int_tuple(0), fx.make_layout((R, N), (0, 1)))

    gIdx = fx.flat_divide(idx_broadcast, (TileM, TileN))[None, None, bid_m, bid_n]
    gDst = fx.flat_divide(Dst, (TileM, TileN))[None, None, bid_m, bid_n]
    cCol = fx.flat_divide(Col, (TileM, TileN))[None, None, bid_m, bid_n]

    thr_gIdx = thr_copy.partition_S(gIdx)[(0, None), None, None]
    thr_cCol = thr_copy.partition_S(cCol)[(0, None), None, None]
    thr_gDst = thr_copy.partition_D(gDst)

    thr_rIdx = fx.make_fragment_like(thr_gIdx)
    thr_rDst = fx.make_fragment_like(thr_gDst)

    fx.copy(atom_i32, thr_gIdx, thr_rIdx)

    # Summing the row index offset and column index gives the global offset.
    thr_rIdx.store(thr_rIdx.load() * N + thr_cCol.load())
    fx.gather(atom_f32, fx.get_iter(Src), thr_rIdx, thr_rDst)

    fx.copy(atom_f32, thr_rDst, thr_gDst)


@flyc.kernel
def scatter_kernel(
    Src: fx.Tensor,  # (R, N) f32
    Dst: fx.Tensor,  # (M, N) f32
    Idx: fx.Tensor,  # (R,) i32 row indices
):
    tid = fx.thread_idx.x
    bid_m, bid_n, _ = fx.block_idx

    atom_f32 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

    thr_copy = fx.make_tiled_copy_tv(
        atom_f32,
        fx.make_layout((TileM, TileN // 4), (TileN // 4, 1)),
        fx.make_layout((1, 4), (1, 1)),
    ).get_slice(tid)

    R, N = Src.shape.unpack()

    # Broadcast index tensor to a per-copy offset tensor.
    idx_broadcast = fx.make_view(fx.get_iter(Idx), fx.make_layout((R, N), (1, 0)))
    # Coordinate tensors for column indices inside the current tile.
    Col = fx.make_view(fx.make_int_tuple(0), fx.make_layout((R, N), (0, 1)))

    gIdx = fx.flat_divide(idx_broadcast, (TileM, TileN))[None, None, bid_m, bid_n]
    gSrc = fx.flat_divide(Src, (TileM, TileN))[None, None, bid_m, bid_n]
    cCol = fx.flat_divide(Col, (TileM, TileN))[None, None, bid_m, bid_n]

    thr_gIdx = thr_copy.partition_S(gIdx)[(0, None), None, None]
    thr_cCol = thr_copy.partition_S(cCol)[(0, None), None, None]
    thr_gSrc = thr_copy.partition_S(gSrc)

    thr_rIdx = fx.make_fragment_like(thr_gIdx)
    thr_rSrc = fx.make_fragment_like(thr_gSrc)

    fx.copy(atom_f32, thr_gSrc, thr_rSrc)
    fx.copy(atom_i32, thr_gIdx, thr_rIdx)

    # Summing the row index offset and column index gives the global offset.
    thr_rIdx.store(thr_rIdx.load() * N + thr_cCol.load())
    fx.scatter(atom_f32, thr_rSrc, fx.get_iter(Dst), thr_rIdx)


@flyc.jit
def gather_jit(
    Src: fx.Tensor,
    Dst: fx.Tensor,
    Idx: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    R, N = Dst.shape.unpack()
    gather_kernel(Src, Dst, Idx).launch(
        grid=(R // TileM, N // TileN, 1), block=[TileM * (TileN // 4), 1, 1], stream=stream
    )


@flyc.jit
def scatter_jit(
    Src: fx.Tensor,
    Dst: fx.Tensor,
    Idx: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    R, N = Src.shape.unpack()
    scatter_kernel(Src, Dst, Idx).launch(
        grid=(R // TileM, N // TileN, 1), block=[TileM * (TileN // 4), 1, 1], stream=stream
    )


if __name__ == "__main__":
    # The kernels launch complete tiles only, so R and N should divide TileM/TileN.
    dev = torch.device("cuda")
    M, R, N = 1024, 512, 2048

    # ---- gather: Dst[r, :] = Src[Idx[r], :] ----
    src = torch.randn(M, N, device=dev, dtype=torch.float32)
    dst = torch.empty(R, N, device=dev, dtype=torch.float32)
    idx = torch.randperm(M, device=dev)[:R].to(torch.int32)

    gather_jit(src, dst, idx)
    torch.cuda.synchronize()

    # Compare on CPU to avoid stream-sensitive GPU comparison behavior.
    gather_ref = src[idx.long()].cpu()
    assert torch.equal(
        dst.cpu(), gather_ref
    ), f"gather mismatch: {(dst.cpu() != gather_ref).sum().item()} / {dst.numel()} elems differ"
    print("PASS: 2D TiledCopy row gather correct (exact match)")

    # ---- scatter: Dst[Idx[r], :] = Src[r, :] (unique Idx rows, no write conflict) ----
    src2 = torch.randn(R, N, device=dev, dtype=torch.float32)
    dst2 = torch.zeros(M, N, device=dev, dtype=torch.float32)
    idx2 = torch.randperm(M, device=dev)[:R].to(torch.int32)

    scatter_jit(src2, dst2, idx2)
    torch.cuda.synchronize()

    scatter_ref = torch.zeros(M, N, dtype=torch.float32)
    scatter_ref[idx2.long().cpu()] = src2.cpu()
    assert torch.equal(
        dst2.cpu(), scatter_ref
    ), f"scatter mismatch: {(dst2.cpu() != scatter_ref).sum().item()} / {dst2.numel()} elems differ"
    print("PASS: 2D TiledCopy row scatter correct (exact match)")
