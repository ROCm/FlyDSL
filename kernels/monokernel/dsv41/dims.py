# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""DeepSeek-V4.1-Flash widths and a TP rank's share of them: every constant a
kernel derives from the TP size, in one place (a build's ``tp`` picks it).
The arithmetic is architecture-neutral; only the tiles that follow from the
matrix instruction (``ROWS``, ``TILE``, ``HEAD_TILE``) are WMMA's 16 x 16."""

from dataclasses import dataclass

# the model's full widths
HIDDEN = 5120
HC = 4  # hyper-connection streams
Q_RANK, KV_DIM = 1280, 512  # the q latent, the window row
Q_HEADS = 64  # attention query heads
O_GROUPS = 8  # wo_a groups
O_RANK = 1024  # wo_a output rows a group
HEAD_DIM = 512
MOE_INTER = 2304  # a routed expert's intermediate width
SHARED_INTER = 2304  # the shared expert's
MOE_PAD = 128  # the loader pads a routed expert's rank share to a multiple of this
ROWS = 16  # GEMV rows a task: a WMMA's M
TILE = 16  # tokens a GEMV pass takes: a WMMA's N
WOA_ROWS = 32  # a wo_a task's rows: one FP8 group of y
UG_PART = 32  # intermediate columns an ug task part: one FP4 group
HEAD_TILE = 16  # heads an attention WMMA takes (its N)


class ShardError(ValueError):
    """A width the TP size does not divide."""


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


@dataclass(frozen=True)
class Tiles:
    """``n`` items in tiles of ``width``; the last tile ``ragged`` when ``width``
    does not divide ``n`` (its items past ``n`` dead)."""

    n: int
    width: int

    @property
    def count(self) -> int:
        return cdiv(self.n, self.width)

    @property
    def ragged(self) -> bool:
        return self.n % self.width != 0


@dataclass(frozen=True)
class Dims:
    tp: int

    def __post_init__(self):
        if self.tp < 1:
            raise ShardError(f"TP {self.tp}")
        for full, what in (
            (Q_HEADS, "query heads"),
            (O_GROUPS, "wo_a groups"),
            (MOE_INTER, "routed intermediate"),
            (SHARED_INTER, "shared intermediate"),
        ):
            if full % self.tp:
                raise ShardError(f"{what} {full} is not divisible by TP {self.tp}")

    @property
    def heads(self) -> int:
        return Q_HEADS // self.tp

    @property
    def head_tiles(self) -> Tiles:
        return Tiles(self.heads, HEAD_TILE)

    @property
    def groups(self) -> int:
        return O_GROUPS // self.tp

    @property
    def group_k(self) -> int:
        """wo_a's K a group: its heads' attention outputs."""
        return self.heads // self.groups * HEAD_DIM

    @property
    def o_rows(self) -> int:
        return self.groups * O_RANK

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def wqb_tasks(self) -> int:
        return self.heads * HEAD_DIM // ROWS

    @property
    def xo_words(self) -> int:
        return self.heads * HEAD_DIM // 4

    @property
    def yb_words(self) -> int:
        return self.o_rows // 4

    @property
    def inter_real(self) -> int:
        return MOE_INTER // self.tp

    @property
    def inter(self) -> int:
        return padded(self.inter_real, MOE_PAD)

    @property
    def sh_inter(self) -> int:
        return SHARED_INTER // self.tp

    @property
    def shared_tasks(self) -> int:
        return self.sh_inter // ROWS

    @property
    def mid_words(self) -> int:
        return self.inter // 4
