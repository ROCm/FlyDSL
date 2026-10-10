// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_TILED_COPY_HPP
#define FLYDSL_CORE_ALGEBRA_TILED_COPY_HPP

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/Layout.hpp"
#include "flydsl/Core/Algebra/OpaqueValue.hpp"
#include "flydsl/Core/Algebra/Tensor.hpp"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "flydsl/Core/Algebra/LiteralMacro.hpp.inc"

namespace mlir::fly::core {

template <class Cat> struct TiledMMA;

template <class Cat> struct CopyAtom {
public:
  using Layout = core::Layout<Cat>;

  struct Storage;

  CopyAtom(OpaqueValue copyOperation, OpaqueValue copyInternalType, int32_t valueBitWidth,
           Layout thrLayout, Layout thrBitLayoutSrc, Layout thrBitLayoutDst,
           Layout thrBitLayoutRef);

  const OpaqueValue &copyOperation() const;
  const OpaqueValue &copyInternalType() const;
  int32_t valueBitWidth() const;

  const Layout &thrLayout() const;
  const Layout &thrBitLayoutSrc() const;
  const Layout &thrBitLayoutDst() const;
  const Layout &thrBitLayoutRef() const;

  Layout valLayoutSrc() const { return recast_layout(valueBitWidth(), 1, thrBitLayoutSrc()); }
  Layout valLayoutDst() const { return recast_layout(valueBitWidth(), 1, thrBitLayoutDst()); }
  Layout valLayoutRef() const { return recast_layout(valueBitWidth(), 1, thrBitLayoutRef()); }

  auto atomNumThr() const { return size<0>(valLayoutRef()); }
  auto atomNumVal() const { return size<1>(valLayoutRef()); }
  auto numValSrc() const { return size<1>(valLayoutSrc()); }
  auto numValDst() const { return size<1>(valLayoutDst()); }

private:
  std::shared_ptr<const Storage> storage_;
};

template <class Cat> struct TiledCopy {
public:
  using Layout = core::Layout<Cat>;

  struct Storage;

  TiledCopy(CopyAtom<Cat> copyAtom, Layout layoutCopyTV, Tile<Cat> tilerMN);

  const CopyAtom<Cat> &copyAtom() const;

  const Layout &layoutCopyTV() const;
  const Layout &tiledLayoutTV() const;

  const Tile<Cat> &tilerMN() const;

  auto tiledNumThr() const { return size<0>(tiledLayoutTV()); }
  auto tiledNumVal() const { return size<1>(tiledLayoutTV()); }

  template <class TensorOrLayout>
  auto tile2thrfrg(const TensorOrLayout &tensor, const Layout &ref2trg) const;

  template <class TensorOrLayout> auto tidfrg_S(const TensorOrLayout &stensor) const;

  template <class TensorOrLayout> auto tidfrg_D(const TensorOrLayout &dtensor) const;

  template <class TensorOrLayout, class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto partition_S(const TensorOrLayout &stensor, const ThrIdx &thr_idx) const;

  template <class TensorOrLayout, class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto partition_D(const TensorOrLayout &dtensor, const ThrIdx &thr_idx) const;

  template <class TensorOrLayout> auto retile(const TensorOrLayout &tensor) const;

  auto get_layoutS_TV() const;

  auto get_layoutD_TV() const;

  template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto get_slice(const ThrIdx &thr_idx) const;

  template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto get_thread_slice(const ThrIdx &thr_idx) const {
    return get_slice(thr_idx);
  }

private:
  std::shared_ptr<const Storage> storage_;
};

template <class Cat> struct ThrCopy {
public:
  ThrCopy(TiledCopy<Cat> tiledCopy, IntTuple<Cat> thrIdx)
      : tiledCopy_(std::move(tiledCopy)), thr_idx_(std::move(thrIdx)) {}

  const TiledCopy<Cat> &tiledCopy() const { return tiledCopy_; }
  const IntTuple<Cat> &threadIndex() const { return thr_idx_; }

  template <class TensorOrLayout> auto partition_S(const TensorOrLayout &stensor) const;

  template <class TensorOrLayout> auto partition_D(const TensorOrLayout &dtensor) const;

  template <class Engine> auto retile_S(const Tensor<Cat, Engine> &stensor) const;

  template <class Engine> auto retile_D(const Tensor<Cat, Engine> &dtensor) const;

private:
  TiledCopy<Cat> tiledCopy_;
  IntTuple<Cat> thr_idx_;
};

// Debug text only: the opaque copy operation and internal type are not
// printed, so copy objects have no fromString.
template <class Cat> void print(const CopyAtom<Cat> &atom, llvm::raw_ostream &os);
template <class Cat> void print(const TiledCopy<Cat> &copy, llvm::raw_ostream &os);
template <class Cat> void print(const ThrCopy<Cat> &copy, llvm::raw_ostream &os);

template <class Cat, class Tiler>
auto make_tiled_copy_impl(const CopyAtom<Cat> &atom, const Layout<Cat> &layoutCopyTV,
                          const Tiler &tiler);
template <class Cat>
auto make_tiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &thr_layout,
                     const Layout<Cat> &val_layout);
template <class Cat>
auto make_tiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &thr_layout);
template <class Cat> auto make_tiled_copy(const CopyAtom<Cat> &copy_atom);

template <class Cat>
auto make_tiled_copy_A(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma);
template <class Cat>
auto make_tiled_copy_B(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma);
template <class Cat>
auto make_tiled_copy_C(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma);
template <class Cat>
auto make_tiled_copy_C_atom(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma);
template <class Cat>
auto make_cotiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &atom_tv_layout,
                       const Layout<Cat> &data_layout);
template <class Cat>
auto make_tiled_copy_S(const CopyAtom<Cat> &copy_atom, const TiledCopy<Cat> &tiled_copy);
template <class Cat>
auto make_tiled_copy_D(const CopyAtom<Cat> &copy_atom, const TiledCopy<Cat> &tiled_copy);
template <int... I, class Cat> auto tile_size(const TiledCopy<Cat> &copy);
template <class Cat> auto size(const TiledCopy<Cat> &copy);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

template <class Cat> struct TiledCopy<Cat>::Storage {
  Storage(CopyAtom<Cat> copyAtom, Layout layoutCopyTV, Tile<Cat> tilerMN)
      : copyAtom(std::move(copyAtom)), layoutCopyTV(std::move(layoutCopyTV)),
        tilerMN(std::move(tilerMN)) {}

  CopyAtom<Cat> copyAtom;
  Layout layoutCopyTV;
  Tile<Cat> tilerMN;
};

template <class Cat> struct CopyAtom<Cat>::Storage {
  Storage(OpaqueValue copyOperation, OpaqueValue copyInternalType, int32_t valueBitWidth,
          Layout thrLayout, Layout thrBitLayoutSrc, Layout thrBitLayoutDst, Layout thrBitLayoutRef);

  OpaqueValue copyOperation;
  OpaqueValue copyInternalType;
  int32_t valueBitWidth;
  Layout thrLayout;
  Layout thrBitLayoutSrc;
  Layout thrBitLayoutDst;
  Layout thrBitLayoutRef;
};

template <class Cat>
CopyAtom<Cat>::Storage::Storage(OpaqueValue copyOperation, OpaqueValue copyInternalType,
                                int32_t valueBitWidth, typename CopyAtom<Cat>::Layout thrLayout,
                                typename CopyAtom<Cat>::Layout thrBitLayoutSrc,
                                typename CopyAtom<Cat>::Layout thrBitLayoutDst,
                                typename CopyAtom<Cat>::Layout thrBitLayoutRef)
    : copyOperation(std::move(copyOperation)), copyInternalType(std::move(copyInternalType)),
      valueBitWidth(valueBitWidth), thrLayout(std::move(thrLayout)),
      thrBitLayoutSrc(std::move(thrBitLayoutSrc)), thrBitLayoutDst(std::move(thrBitLayoutDst)),
      thrBitLayoutRef(std::move(thrBitLayoutRef)) {}

template <class Cat>
CopyAtom<Cat>::CopyAtom(OpaqueValue copyOperation, OpaqueValue copyInternalType,
                        int32_t valueBitWidth, typename CopyAtom<Cat>::Layout thrLayout,
                        typename CopyAtom<Cat>::Layout thrBitLayoutSrc,
                        typename CopyAtom<Cat>::Layout thrBitLayoutDst,
                        typename CopyAtom<Cat>::Layout thrBitLayoutRef) {
  storage_ = std::make_shared<Storage>(
      std::move(copyOperation), std::move(copyInternalType), valueBitWidth, std::move(thrLayout),
      std::move(thrBitLayoutSrc), std::move(thrBitLayoutDst), std::move(thrBitLayoutRef));
  FLYDSL_CORE_ASSERT(storage_->valueBitWidth > 0);
  FLYDSL_CORE_ASSERT(storage_->thrBitLayoutSrc.rank() >= 2 &&
                     storage_->thrBitLayoutDst.rank() >= 2 &&
                     storage_->thrBitLayoutRef.rank() >= 2);

  FLYDSL_CORE_ASSERT(size<0>(valLayoutSrc()) == storage_->thrLayout.size());
  FLYDSL_CORE_ASSERT(size<0>(valLayoutDst()) == storage_->thrLayout.size());
  FLYDSL_CORE_ASSERT(size<0>(valLayoutRef()) == storage_->thrLayout.size());
}

template <class Cat> const OpaqueValue &CopyAtom<Cat>::copyOperation() const {
  return storage_->copyOperation;
}

template <class Cat> const OpaqueValue &CopyAtom<Cat>::copyInternalType() const {
  return storage_->copyInternalType;
}

template <class Cat> int32_t CopyAtom<Cat>::valueBitWidth() const {
  return storage_->valueBitWidth;
}

template <class Cat> const typename CopyAtom<Cat>::Layout &CopyAtom<Cat>::thrLayout() const {
  return storage_->thrLayout;
}

template <class Cat> const typename CopyAtom<Cat>::Layout &CopyAtom<Cat>::thrBitLayoutSrc() const {
  return storage_->thrBitLayoutSrc;
}

template <class Cat> const typename CopyAtom<Cat>::Layout &CopyAtom<Cat>::thrBitLayoutDst() const {
  return storage_->thrBitLayoutDst;
}

template <class Cat> const typename CopyAtom<Cat>::Layout &CopyAtom<Cat>::thrBitLayoutRef() const {
  return storage_->thrBitLayoutRef;
}

template <class Cat>
TiledCopy<Cat>::TiledCopy(CopyAtom<Cat> copyAtom, typename TiledCopy<Cat>::Layout layoutCopyTV,
                          Tile<Cat> tilerMN) {
  storage_ =
      std::make_shared<Storage>(std::move(copyAtom), std::move(layoutCopyTV), std::move(tilerMN));

  auto tiledThreads = tiledNumThr();
  auto tiledValues = tiledNumVal();
  auto atomThreads = storage_->copyAtom.atomNumThr();
  auto atomValues = storage_->copyAtom.atomNumVal();

  FLYDSL_CORE_ASSERT(tiledThreads.staticValue() % atomThreads.staticValue() == 0);
  FLYDSL_CORE_ASSERT(tiledValues.staticValue() % atomValues.staticValue() == 0);
}

template <class Cat>
template <class TensorOrLayout>
auto TiledCopy<Cat>::tile2thrfrg(const TensorOrLayout &tensor, const Layout &ref2trg) const {
  auto atom_layout_TV =
      zipped_divide(tiledLayoutTV(), make_tuple(copyAtom().atomNumThr(), copyAtom().atomNumVal()));
  auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);

  auto thrval2mn = coalesce(zip(trg_layout_TV), make_tuple(_1, make_tuple(_1, _1)));

  auto tv_tensor = tensor.compose(thrval2mn, _);

  return tv_tensor(make_tuple(_, _), _);
}

template <class Cat>
template <class TensorOrLayout>
auto TiledCopy<Cat>::tidfrg_S(const TensorOrLayout &stensor) const {
  auto tiled = zipped_divide(stensor, tilerMN());
  if (tiled.isError())
    return detail::tagError(tiled, AlgebraOp::TiledCopyPartition);

  return tile2thrfrg(tiled,
                     right_inverse(copyAtom().valLayoutRef()).compose(copyAtom().valLayoutSrc()));
}

template <class Cat>
template <class TensorOrLayout>
auto TiledCopy<Cat>::tidfrg_D(const TensorOrLayout &dtensor) const {
  auto tiled = zipped_divide(dtensor, tilerMN());
  if (tiled.isError())
    return detail::tagError(tiled, AlgebraOp::TiledCopyPartition);
  return tile2thrfrg(tiled,
                     right_inverse(copyAtom().valLayoutRef()).compose(copyAtom().valLayoutDst()));
}

template <class Cat>
template <class TensorOrLayout, class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int>>
auto TiledCopy<Cat>::partition_S(const TensorOrLayout &stensor, const ThrIdx &thr_idx) const {
  auto thr_tensor = tidfrg_S(stensor);
  if (thr_tensor.isError())
    return thr_tensor;
  return thr_tensor(make_tuple(thr_idx, _, repeat(_, rank(stensor))));
}

template <class Cat>
template <class TensorOrLayout, class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int>>
auto TiledCopy<Cat>::partition_D(const TensorOrLayout &dtensor, const ThrIdx &thr_idx) const {
  auto thr_tensor = tidfrg_D(dtensor);
  if (thr_tensor.isError())
    return thr_tensor;
  return thr_tensor(make_tuple(thr_idx, _, repeat(_, rank(dtensor))));
}

template <class Cat>
template <class TensorOrLayout>
auto TiledCopy<Cat>::retile(const TensorOrLayout &tensor) const {
  if (tensor.isError())
    return detail::tagError(tensor, AlgebraOp::TiledCopyRetile);
  auto V = size<0>(tensor);

  auto upcastFactor = tiledNumThr() * V;
  if (!upcastFactor.isSInt())
    return detail::errorLike(
        tensor, ErrorInfo(ErrorCode::ExpectedStaticOperand).withFrame(AlgebraOp::TiledCopyRetile));

  auto frg_layout_mn = upcast(upcastFactor.staticValue(),
                              right_inverse(tiledLayoutTV()).with_shape(tilerMN().shape()));

  auto frg_layout_v = zipped_divide(logical_product(make_layout(V), right_inverse(frg_layout_mn)),
                                    make_layout(copyAtom().atomNumVal()));

  auto t_tensor = zipped_divide(tensor, prepend(product_each(frg_layout_mn.shape()), V));

  auto v_tensor = t_tensor.compose(frg_layout_v, _);

  return v_tensor(_, append(_0, _, tensor.rank()));
}

template <class Cat> auto TiledCopy<Cat>::get_layoutS_TV() const {
  auto ref_S = make_layout(make_tuple(tilerMN().shape(), _1));
  return tile2thrfrg(
      ref_S, right_inverse(copyAtom().valLayoutRef()).compose(copyAtom().valLayoutSrc()))(_, _, _0);
}

template <class Cat> auto TiledCopy<Cat>::get_layoutD_TV() const {
  auto ref_D = make_layout(make_tuple(tilerMN().shape(), _1));
  return tile2thrfrg(
      ref_D, right_inverse(copyAtom().valLayoutRef()).compose(copyAtom().valLayoutDst()))(_, _, _0);
}

template <class Cat> const CopyAtom<Cat> &TiledCopy<Cat>::copyAtom() const {
  return storage_->copyAtom;
}

template <class Cat> const typename TiledCopy<Cat>::Layout &TiledCopy<Cat>::layoutCopyTV() const {
  return storage_->layoutCopyTV;
}

template <class Cat> const typename TiledCopy<Cat>::Layout &TiledCopy<Cat>::tiledLayoutTV() const {
  return storage_->layoutCopyTV;
}

template <class Cat> const Tile<Cat> &TiledCopy<Cat>::tilerMN() const { return storage_->tilerMN; }

template <class Cat>
template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int>>
auto TiledCopy<Cat>::get_slice(const ThrIdx &thr_idx) const {
  static_assert(std::is_same_v<Cat, category_of<ThrIdx>>);
  return ThrCopy<Cat>(*this, thr_idx.asRef());
}

template <class Cat>
template <class TensorOrLayout>
auto ThrCopy<Cat>::partition_S(const TensorOrLayout &stensor) const {
  auto thr_tensor = tiledCopy_.tidfrg_S(stensor);
  if (thr_tensor.isError())
    return thr_tensor;
  return thr_tensor(thr_idx_, _, repeat(_, rank(stensor)));
}

template <class Cat>
template <class TensorOrLayout>
auto ThrCopy<Cat>::partition_D(const TensorOrLayout &dtensor) const {
  auto thr_tensor = tiledCopy_.tidfrg_D(dtensor);
  if (thr_tensor.isError())
    return thr_tensor;
  return thr_tensor(thr_idx_, _, repeat(_, rank(dtensor)));
}

template <class Cat>
template <class Engine>
auto ThrCopy<Cat>::retile_S(const Tensor<Cat, Engine> &stensor) const {
  return tiledCopy_.retile(stensor);
}

template <class Cat>
template <class Engine>
auto ThrCopy<Cat>::retile_D(const Tensor<Cat, Engine> &dtensor) const {
  return tiledCopy_.retile(dtensor);
}

template <class Cat, class Tiler>
auto make_tiled_copy_impl(const CopyAtom<Cat> &atom, const Layout<Cat> &layoutCopyTV,
                          const Tiler &tiler) {
  if constexpr (is_int_tuple_v<Tiler>)
    return TiledCopy<Cat>(atom, layoutCopyTV, shape_to_tile(tiler));
  else
    return TiledCopy<Cat>(atom, layoutCopyTV, detail::toTile(tiler));
}

template <class Cat>
auto make_tiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &thr_layout,
                     const Layout<Cat> &val_layout) {
  auto layoutMN = raked_product(thr_layout, val_layout);

  auto layoutTV =
      right_inverse(layoutMN).with_shape(make_tuple(thr_layout.size(), val_layout.size()));
  auto tiler = product_each(layoutMN.shape());

  return make_tiled_copy_impl(copy_atom, layoutTV, tiler);
}

template <class Cat>
auto make_tiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &thr_layout) {
  auto valLayout = make_layout(IntTuple<Cat>::getOne());
  return make_tiled_copy(copy_atom, thr_layout, valLayout);
}

template <class Cat> auto make_tiled_copy(const CopyAtom<Cat> &copy_atom) {
  auto thrLayout = make_layout(IntTuple<Cat>::getOne());
  return make_tiled_copy(copy_atom, thrLayout);
}

template <class Cat>
auto make_tiled_copy_A(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma) {
  return make_tiled_copy_impl(
      copy_atom, mma.get_layoutA_TV(),
      make_tuple(mma.template tile_size_mnk<0>(), mma.template tile_size_mnk<2>()));
}

template <class Cat>
auto make_tiled_copy_B(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma) {
  return make_tiled_copy_impl(
      copy_atom, mma.get_layoutB_TV(),
      make_tuple(mma.template tile_size_mnk<1>(), mma.template tile_size_mnk<2>()));
}

template <class Cat>
auto make_tiled_copy_C(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma) {
  return make_tiled_copy_impl(
      copy_atom, mma.get_layoutC_TV(),
      make_tuple(mma.template tile_size_mnk<0>(), mma.template tile_size_mnk<1>()));
}

template <class Cat>
auto make_tiled_copy_C_atom(const CopyAtom<Cat> &copy_atom, const TiledMMA<Cat> &mma) {
  auto layoutC_TV = mma.get_layoutC_TV();
  auto copy_V = copy_atom.numValSrc();
  // The atom must not take more values than a thread holds in C; otherwise the
  // extra values would map onto the same element.
  FLYDSL_CORE_ASSERT(copy_V.staticValue() <= size<1>(layoutC_TV).staticValue());

  auto layout_TV = composition(layoutC_TV, make_layout(make_tuple(size<0>(layoutC_TV), copy_V)));

  auto mma_tiler = make_tuple(mma.template tile_size_mnk<0>(), mma.template tile_size_mnk<1>());

  auto mma_zeros = repeat_like(mma_tiler, _0);

  llvm::SmallVector<Tile<Cat>, 8> tiler_modes;
  auto mma_tiler_rank = mma_tiler.rank();

  for (auto i = int32_t{0}; i < mma_tiler_rank; ++i) {
    auto stride = replace(mma_zeros, i, _1);
    tiler_modes.emplace_back(
        filter(composition(make_layout(mma_tiler, std::move(stride)), layout_TV)));
  }
  auto tiler = Tile<Cat>::getTuple(tiler_modes);

  auto tile2mma = composition(make_layout(std::move(mma_tiler)), tiler);

  auto layout_tv = composition(left_inverse(tile2mma), layout_TV);

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

template <class Cat>
auto make_cotiled_copy(const CopyAtom<Cat> &copy_atom, const Layout<Cat> &atom_tv_layout,
                       const Layout<Cat> &data_layout) {

  auto unit_zero = make_layout(_1, _0);

  auto inv_data_layout = make_layout(left_inverse(data_layout), unit_zero);

  auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);

  FLYDSL_CORE_ASSERT(
      coalesce(composition(make_layout(data_layout, unit_zero), layout<1>(layout_tv_data))) ==
      coalesce(layout<1>(atom_tv_layout)));

  auto flat_data_shape = product_each(shape(data_layout));
  auto flat_data_zeros = repeat_like(flat_data_shape, _0);

  llvm::SmallVector<Tile<Cat>, 8> tiler_modes;
  auto flat_data_rank = flat_data_shape.rank();
  for (auto i = int32_t{0}; i < flat_data_rank; ++i) {
    auto stride = replace(flat_data_zeros, i, _1);
    tiler_modes.emplace_back(
        filter(composition(make_layout(flat_data_shape, std::move(stride)), layout_tv_data)));
  }
  auto tiler = Tile<Cat>::getTuple(tiler_modes);

  auto tile2data = composition(make_layout(std::move(flat_data_shape)), tiler);

  auto layout_tv = composition(left_inverse(tile2data), layout_tv_data);
  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

template <class Cat>
auto make_tiled_copy_S(const CopyAtom<Cat> &copy_atom, const TiledCopy<Cat> &tiled_copy) {
  return make_tiled_copy_impl(copy_atom, tiled_copy.get_layoutS_TV(), tiled_copy.tilerMN());
}

template <class Cat>
auto make_tiled_copy_D(const CopyAtom<Cat> &copy_atom, const TiledCopy<Cat> &tiled_copy) {
  return make_tiled_copy_impl(copy_atom, tiled_copy.get_layoutD_TV(), tiled_copy.tilerMN());
}

template <int... I, class Cat> auto tile_size(const TiledCopy<Cat> &copy) {
  if constexpr (sizeof...(I) == 0) {
    return product(copy.tilerMN().shape());
  } else {
    return size(get<I...>(copy.tilerMN()));
  }
}

template <class Cat> auto size(const TiledCopy<Cat> &copy) { return copy.tiledNumThr(); }

template <class Cat> void print(const CopyAtom<Cat> &atom, llvm::raw_ostream &os) {
  os << "copy_atom<bits=" << atom.valueBitWidth() << ",thr_layout=";
  print(atom.thrLayout(), os);
  os << ",thr_bit_src=";
  print(atom.thrBitLayoutSrc(), os);
  os << ",thr_bit_dst=";
  print(atom.thrBitLayoutDst(), os);
  os << ",thr_bit_ref=";
  print(atom.thrBitLayoutRef(), os);
  os << '>';
}

template <class Cat> void print(const TiledCopy<Cat> &copy, llvm::raw_ostream &os) {
  os << "tiled_copy<";
  print(copy.copyAtom(), os);
  os << ",layout_tv=";
  print(copy.layoutCopyTV(), os);
  os << ",tiler_mn=";
  print(copy.tilerMN(), os);
  os << '>';
}

template <class Cat> void print(const ThrCopy<Cat> &copy, llvm::raw_ostream &os) {
  os << "thr_copy<";
  print(copy.tiledCopy(), os);
  os << ",thr_idx=";
  print(copy.threadIndex(), os);
  os << '>';
}

} // namespace mlir::fly::core

#include "flydsl/Core/Algebra/LiteralMacro.hpp.inc" // undef _, _0, _1

#endif // FLYDSL_CORE_ALGEBRA_TILED_COPY_HPP
