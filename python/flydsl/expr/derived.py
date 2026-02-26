from .._mlir import ir
from .._mlir.dialects._fly_enum_gen import MmaOperand
from .primitive import *
from .typing import Layout, Tensor

__all__ = [
    # Tiled Operation
    "CopyAtom",
    "MmaAtom",
    "TiledCopy",
    "TiledMma",
    "ThrCopy",
    "ThrMma",
    "make_layout_tv",
    "make_tiled_copy",
]


class Atom:
    def __init__(self, value: ir.Value):
        self.value = value
        self.atom_ty = self.value.type

    @classmethod
    def __new_from_ir_values__(cls, values):
        return cls(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class CopyAtom:
    def __init__(self, value: ir.Value):
        self.value = value
        self.atom_ty = self.value.type

    @classmethod
    def __new_from_ir_values__(cls, values):
        return cls(values[0])

    def __extract_ir_values__(self):
        return [self.value]

    def __str__(self):
        return f"CopyAtom({self.atom_ty})"

    @property
    def thr_layout(self):
        return static(self.atom_ty.thr_layout)

    @property
    def tv_layout_src(self):
        return static(self.atom_ty.tv_layout_src)

    @property
    def tv_layout_dst(self):
        return static(self.atom_ty.tv_layout_dst)

    @property
    def tv_layout_ref(self):
        return static(self.atom_ty.tv_layout_ref)


class MmaAtom(Atom):
    def __str__(self):
        return f"MmaAtom({self.atom_ty})"

    @property
    def thr_layout(self):
        return static(self.atom_ty.thr_layout)

    @property
    def shape_mnk(self):
        return static(self.atom_ty.shape_mnk)

    @property
    def tv_layout_A(self):
        return static(self.atom_ty.tv_layout_a)

    @property
    def tv_layout_B(self):
        return static(self.atom_ty.tv_layout_b)

    @property
    def tv_layout_C(self):
        return static(self.atom_ty.tv_layout_c)


class TiledCopy:
    def __init__(self, value):
        self.value = value
        self.tiled_copy_ty = self.value.type

    @classmethod
    def __new_from_ir_values__(cls, values):
        return cls(values[0])

    def __extract_ir_values__(self):
        return [self.value]

    def __str__(self):
        return f"TiledCopy({self.tiled_copy_ty})"

    def get_slice(self, thr_idx):
        return ThrCopy(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    @property
    def tiled_tv_layout_S(self):
        return static(self.tiled_copy_ty.tiled_tv_layout_src)

    @property
    def tiled_tv_layout_D(self):
        return static(self.tiled_copy_ty.tiled_tv_layout_dst)


class TiledMma:
    def __init__(self, value):
        self.value = value
        self.tiled_mma_ty = self.value.type

    @classmethod
    def __new_from_ir_values__(cls, values):
        return cls(values[0])

    def __extract_ir_values__(self):
        return [self.value]

    def __str__(self):
        return f"TiledMma({self.tiled_mma_ty})"

    def get_slice(self, thr_idx):
        return ThrMma(self, thr_idx)

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    @property
    def tile_size_mnk(self):
        return static(self.tiled_mma_ty.tile_size_mnk)

    @property
    def thr_layout_vmnk(self):
        return static(self.tiled_mma_ty.thr_layout_vmnk)

    @property
    def tiled_tv_layout_A(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_a)

    @property
    def tiled_tv_layout_B(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_b)

    @property
    def tiled_tv_layout_C(self):
        return static(self.tiled_mma_ty.tiled_tv_layout_c)


class ThrCopy(TiledCopy):
    def __init__(self, tiled_copy: TiledCopy, thr_idx):
        super().__init__(tiled_copy.value)
        self.tiled_copy = tiled_copy
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    def partition_S(self, src: Tensor):
        return tiled_copy_partition_src(self.value, src, self._thr_idx_int)

    def partition_D(self, dst: Tensor):
        return tiled_copy_partition_dst(self.value, dst, self._thr_idx_int)

    def retile(self, t: Tensor):
        # return tiled_copy_retile(self.value, t)
        raise NotImplementedError("Retiling is not supported yet")


class ThrMma(TiledMma):
    def __init__(self, tiled_mma: TiledMma, thr_idx):
        super().__init__(tiled_mma.value)
        self.tiled_mma = tiled_mma
        self._thr_idx = thr_idx
        self._thr_idx_int = make_int_tuple(self.thr_idx)

    @property
    def thr_idx(self):
        return self._thr_idx

    def partition_A(self, a: Tensor):
        return tiled_mma_partition(MmaOperand.A, self.tiled_mma.value, a, self._thr_idx_int)

    def partition_B(self, b: Tensor):
        return tiled_mma_partition(MmaOperand.B, self.tiled_mma.value, b, self._thr_idx_int)

    def partition_C(self, c: Tensor):
        return tiled_mma_partition(MmaOperand.C, self.tiled_mma.value, c, self._thr_idx_int)


def make_layout_tv(thr_layout, val_layout, loc=None, ip=None):
    layout_mn = raked_product(thr_layout, val_layout)
    thr_size = size(thr_layout)
    val_size = size(val_layout)
    tmp = make_layout((thr_size, val_size), (1, thr_size))

    layout_tv = composition(right_inverse(layout_mn), tmp)

    tiler_mn = int_tuple_product_each(get_shape(layout_mn))
    return (tiler_mn, layout_tv)


def make_tiled_copy_A(copy_atom, tiled_mma):
    pass


def make_tiled_copy_B(copy_atom, tiled_mma):
    pass


def make_tiled_copy_C(copy_atom, tiled_mma):
    pass
