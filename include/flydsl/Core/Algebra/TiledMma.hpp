// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_TILED_MMA_HPP
#define FLYDSL_CORE_ALGEBRA_TILED_MMA_HPP

#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/Layout.hpp"
#include "flydsl/Core/Algebra/OpaqueValue.hpp"
#include "flydsl/Core/Algebra/Tensor.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include "flydsl/Core/Algebra/LiteralMacro.hpp.inc"

namespace mlir::fly::core {

template <class Cat> struct ThrMMA;

struct MMAValueTypeIdentities {
  OpaqueValue valTypeA;
  OpaqueValue valTypeB;
  OpaqueValue valTypeC;
  OpaqueValue valTypeD;
};

template <class Cat> struct MMAAtom {
public:
  struct Storage;

  MMAAtom(OpaqueValue mmaOperation, Layout<Cat> thrLayout, IntTuple<Cat> shapeMNK,
          MMAValueTypeIdentities valueTypes, Layout<Cat> thrValLayoutA, Layout<Cat> thrValLayoutB,
          Layout<Cat> thrValLayoutC);

  const OpaqueValue &mmaOperation() const;
  const MMAValueTypeIdentities &valueTypes() const;

  IntTupleRef<Cat> shapeMNK() const FLYDSL_CORE_LIFETIMEBOUND;
  const Layout<Cat> &thrLayout() const;

  const OpaqueValue &valTypeA() const { return valueTypes().valTypeA; }
  const OpaqueValue &valTypeB() const { return valueTypes().valTypeB; }
  const OpaqueValue &valTypeC() const { return valueTypes().valTypeC; }
  const OpaqueValue &valTypeD() const { return valueTypes().valTypeD; }

  const Layout<Cat> &thrValLayoutA() const;
  const Layout<Cat> &thrValLayoutB() const;
  const Layout<Cat> &thrValLayoutC() const;

private:
  std::shared_ptr<const Storage> storage_;
};

template <class Cat> struct TiledMMA {
public:
  struct Storage;

  TiledMMA(MMAAtom<Cat> mmaAtom, Layout<Cat> atomLayoutMNK, Tile<Cat> permutationMNK);

  const MMAAtom<Cat> &mmaAtom() const;
  const Layout<Cat> &atomLayoutMNK() const;

  const Layout<Cat> &get_thr_layout_vmnk() const;

  const Tile<Cat> &permutationMNK() const;
  Layout<Cat> permutation_mnk(int32_t i) const;
  template <int I> Layout<Cat> permutation_mnk() const;

  auto tile_size_mnk(int32_t i) const { return permutation_mnk(i).size(); }
  template <int I> auto tile_size_mnk() const;

  template <class TensorOrLayout> auto thrfrg_A(const TensorOrLayout &atensor) const;

  template <class TensorOrLayout> auto thrfrg_B(const TensorOrLayout &btensor) const;

  template <class TensorOrLayout> auto thrfrg_C(const TensorOrLayout &ctensor) const;

  template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto get_slice(const ThrIdx &thr_idx) const;

  template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int> = 0>
  auto get_thread_slice(const ThrIdx &thr_idx) const {
    return get_slice(thr_idx);
  }

  Layout<Cat> get_layoutC_TV() const;

  Layout<Cat> get_layoutA_TV() const;

  Layout<Cat> get_layoutB_TV() const;

private:
  std::shared_ptr<const Storage> storage_;
};

template <class Cat> struct ThrMMA : public TiledMMA<Cat> {
public:
  ThrMMA(TiledMMA<Cat> tiledMMA, IntTuple<Cat> thr_vmnk)
      : TiledMMA<Cat>(std::move(tiledMMA)), thr_vmnk_(std::move(thr_vmnk)) {}

  const TiledMMA<Cat> &tiledMMA() const { return *this; }
  const IntTuple<Cat> &threadCoord() const { return thr_vmnk_; }

  template <class Engine> auto partition_C(const Tensor<Cat, Engine> &ctensor) const;

  template <class Engine> auto partition_A(const Tensor<Cat, Engine> &atensor) const;

  template <class Engine> auto partition_B(const Tensor<Cat, Engine> &btensor) const;

  IntTuple<Cat> thr_vmnk_;
};

// Debug text only: the opaque MMA operation and value types are not printed,
// so MMA objects have no fromString.
template <class Cat> void print(const MMAAtom<Cat> &atom, llvm::raw_ostream &os);
template <class Cat> void print(const TiledMMA<Cat> &mma, llvm::raw_ostream &os);
template <class Cat> void print(const ThrMMA<Cat> &mma, llvm::raw_ostream &os);

template <class Cat>
auto make_tiled_mma(const MMAAtom<Cat> &mma_atom, const Layout<Cat> &thr_layout,
                    const Tile<Cat> &permutations);
template <class Cat>
auto make_tiled_mma(const MMAAtom<Cat> &mma_atom, const Layout<Cat> &thr_layout);

template <class Cat> auto make_tiled_mma(const MMAAtom<Cat> &mma_atom);

template <int I, class Cat> auto tile_size(const TiledMMA<Cat> &mma);
template <class Cat> auto tile_shape(const TiledMMA<Cat> &mma);
template <int... I, class Cat> auto size(const TiledMMA<Cat> &mma);
template <int... I, class Cat> auto thr_size(const TiledMMA<Cat> &mma);

template <class Cat, class Shape>
IntTuple<Cat> partition_shape_C(const TiledMMA<Cat> &mma, const Shape &shape_MN);
template <class Cat, class Shape>
IntTuple<Cat> partition_shape_A(const TiledMMA<Cat> &mma, const Shape &shape_MK);
template <class Cat, class Shape>
IntTuple<Cat> partition_shape_B(const TiledMMA<Cat> &mma, const Shape &shape_NK);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

template <class Cat> struct MMAAtom<Cat>::Storage {
  OpaqueValue mmaOperation;
  Layout<Cat> thrLayout;
  IntTuple<Cat> shapeMNK;
  MMAValueTypeIdentities valueTypes;
  Layout<Cat> thrValLayoutA;
  Layout<Cat> thrValLayoutB;
  Layout<Cat> thrValLayoutC;

  Storage(OpaqueValue mmaOperation, Layout<Cat> thrLayout, IntTuple<Cat> shapeMNK,
          MMAValueTypeIdentities valueTypes, Layout<Cat> thrValLayoutA, Layout<Cat> thrValLayoutB,
          Layout<Cat> thrValLayoutC)
      : mmaOperation(std::move(mmaOperation)), thrLayout(std::move(thrLayout)),
        shapeMNK(std::move(shapeMNK)), valueTypes(std::move(valueTypes)),
        thrValLayoutA(std::move(thrValLayoutA)), thrValLayoutB(std::move(thrValLayoutB)),
        thrValLayoutC(std::move(thrValLayoutC)) {}
};

template <class Cat> struct TiledMMA<Cat>::Storage {
  Storage(MMAAtom<Cat> mmaAtom, Layout<Cat> atomLayoutMNK, Tile<Cat> permutationMNK)
      : mmaAtom(std::move(mmaAtom)), atomLayoutMNK(std::move(atomLayoutMNK)),
        permutationMNK(std::move(permutationMNK)),
        thrLayoutVMNK(tiled_product(this->mmaAtom.thrLayout(), this->atomLayoutMNK)) {}

  MMAAtom<Cat> mmaAtom;
  Layout<Cat> atomLayoutMNK;
  Tile<Cat> permutationMNK;
  Layout<Cat> thrLayoutVMNK;
};

template <class Cat>
MMAAtom<Cat>::MMAAtom(OpaqueValue mmaOperation, Layout<Cat> thrLayout, IntTuple<Cat> shapeMNK,
                      MMAValueTypeIdentities valueTypes, Layout<Cat> thrValLayoutA,
                      Layout<Cat> thrValLayoutB, Layout<Cat> thrValLayoutC) {
  storage_ = std::make_shared<Storage>(
      std::move(mmaOperation), std::move(thrLayout), std::move(shapeMNK), std::move(valueTypes),
      std::move(thrValLayoutA), std::move(thrValLayoutB), std::move(thrValLayoutC));
  FLYDSL_CORE_ASSERT(storage_->shapeMNK.rank() == 3);
  FLYDSL_CORE_ASSERT(storage_->thrValLayoutA.rank() >= 2 && storage_->thrValLayoutB.rank() >= 2 &&
                     storage_->thrValLayoutC.rank() >= 2);
  FLYDSL_CORE_ASSERT(size<0>(storage_->thrValLayoutA) == storage_->thrLayout.size() &&
                     size<0>(storage_->thrValLayoutB) == storage_->thrLayout.size() &&
                     size<0>(storage_->thrValLayoutC) == storage_->thrLayout.size());
}

template <class Cat> const OpaqueValue &MMAAtom<Cat>::mmaOperation() const {
  return storage_->mmaOperation;
}

template <class Cat> const Layout<Cat> &MMAAtom<Cat>::thrLayout() const {
  return storage_->thrLayout;
}

template <class Cat> IntTupleRef<Cat> MMAAtom<Cat>::shapeMNK() const {
  return storage_->shapeMNK.asRef();
}

template <class Cat> const MMAValueTypeIdentities &MMAAtom<Cat>::valueTypes() const {
  return storage_->valueTypes;
}

template <class Cat> const Layout<Cat> &MMAAtom<Cat>::thrValLayoutA() const {
  return storage_->thrValLayoutA;
}

template <class Cat> const Layout<Cat> &MMAAtom<Cat>::thrValLayoutB() const {
  return storage_->thrValLayoutB;
}

template <class Cat> const Layout<Cat> &MMAAtom<Cat>::thrValLayoutC() const {
  return storage_->thrValLayoutC;
}

template <class Cat>
TiledMMA<Cat>::TiledMMA(MMAAtom<Cat> mmaAtom, Layout<Cat> atomLayoutMNK, Tile<Cat> permutationMNK) {
  storage_ = std::make_shared<Storage>(std::move(mmaAtom), std::move(atomLayoutMNK),
                                       std::move(permutationMNK));
  FLYDSL_CORE_ASSERT(storage_->atomLayoutMNK.rank() == 3);
  FLYDSL_CORE_ASSERT(storage_->permutationMNK.rank() == 3);
  FLYDSL_CORE_ASSERT(storage_->permutationMNK.isStatic());
}

template <class Cat> const MMAAtom<Cat> &TiledMMA<Cat>::mmaAtom() const {
  return storage_->mmaAtom;
}

template <class Cat> const Layout<Cat> &TiledMMA<Cat>::atomLayoutMNK() const {
  return storage_->atomLayoutMNK;
}

template <class Cat> const Tile<Cat> &TiledMMA<Cat>::permutationMNK() const {
  return storage_->permutationMNK;
}

template <class Cat> const Layout<Cat> &TiledMMA<Cat>::get_thr_layout_vmnk() const {
  return storage_->thrLayoutVMNK;
}

template <class Cat> Layout<Cat> TiledMMA<Cat>::permutation_mnk(int32_t i) const {
  FLYDSL_CORE_ASSERT(i >= 0 && i < 3);
  auto permutation = permutationMNK().at(i);
  if (permutation.isLayout())
    return permutation.getLayout();
  if (permutation.isScalar())
    return make_layout(IntTuple<Cat>::fromLeaf(permutation.getScalar()));
  // A mode is a layout, a scalar, or `_`; a nested tile is not a permutation.
  if (!permutation.isNone())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::TiledMmaPermutation, i));
  auto extent = product(mmaAtom().shapeMNK().at(i)) * get(get_thr_layout_vmnk(), i + 1).size();
  return make_layout(std::move(extent));
}

template <class Cat> template <int I> Layout<Cat> TiledMMA<Cat>::permutation_mnk() const {
  static_assert(I >= 0 && I < 3);
  auto permutation = get<I>(permutationMNK());
  if (permutation.isLayout())
    return permutation.getLayout();
  if (permutation.isScalar())
    return make_layout(IntTuple<Cat>::fromLeaf(permutation.getScalar()));
  if (!permutation.isNone())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::TiledMmaPermutation, I));
  return make_layout(size<I>(mmaAtom().shapeMNK()) * size<I + 1>(get_thr_layout_vmnk()));
}

template <class Cat> template <int I> auto TiledMMA<Cat>::tile_size_mnk() const {
  static_assert(I >= 0 && I < 3);
  return permutation_mnk<I>().size();
}

template <class Cat>
template <class TensorOrLayout>
auto TiledMMA<Cat>::thrfrg_C(const TensorOrLayout &ctensor) const {
  auto t_tile = make_tile(permutation_mnk<0>(), permutation_mnk<1>());
  auto t_tensor = logical_divide(ctensor, t_tile);

  auto c_tile = make_tile(make_layout(size<0>(mmaAtom().shapeMNK())),
                          make_layout(size<1>(mmaAtom().shapeMNK())));
  auto c_tensor = zipped_divide(t_tensor, c_tile);

  auto tv_tensor = c_tensor.compose(mmaAtom().thrValLayoutC(), _);

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<2>(thr_layout_vmnk))));
  return detail::tagError(zipped_divide(tv_tensor, thr_tile), AlgebraOp::TiledMmaPartition);
}

template <class Cat>
template <class TensorOrLayout>
auto TiledMMA<Cat>::thrfrg_A(const TensorOrLayout &atensor) const {
  auto t_tile = make_tile(permutation_mnk<0>(), permutation_mnk<2>());
  auto t_tensor = logical_divide(atensor, t_tile);

  auto a_tile = make_tile(make_layout(size<0>(mmaAtom().shapeMNK())),
                          make_layout(size<2>(mmaAtom().shapeMNK())));
  auto a_tensor = zipped_divide(t_tensor, a_tile);

  auto tv_tensor = a_tensor.compose(mmaAtom().thrValLayoutA(), _);

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return detail::tagError(zipped_divide(tv_tensor, thr_tile), AlgebraOp::TiledMmaPartition);
}

template <class Cat>
template <class TensorOrLayout>
auto TiledMMA<Cat>::thrfrg_B(const TensorOrLayout &btensor) const {
  auto t_tile = make_tile(permutation_mnk<1>(), permutation_mnk<2>());
  auto t_tensor = logical_divide(btensor, t_tile);

  auto b_tile = make_tile(make_layout(size<1>(mmaAtom().shapeMNK())),
                          make_layout(size<2>(mmaAtom().shapeMNK())));
  auto b_tensor = zipped_divide(t_tensor, b_tile);

  auto tv_tensor = b_tensor.compose(mmaAtom().thrValLayoutB(), _);

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return detail::tagError(zipped_divide(tv_tensor, thr_tile), AlgebraOp::TiledMmaPartition);
}

template <class Cat> Layout<Cat> TiledMMA<Cat>::get_layoutC_TV() const {
  auto ref_C = make_layout(make_tuple(tile_size_mnk<0>(), tile_size_mnk<1>()));

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto linear = make_layout(make_tuple(thr_layout_vmnk.size(), _1), make_tuple(_1, _0));
  auto thridx_2_thrid =
      composition(linear, right_inverse(make_layout(thr_layout_vmnk, complement(thr_layout_vmnk))));
  return thrfrg_C(ref_C).compose(thridx_2_thrid, _);
}

template <class Cat> Layout<Cat> TiledMMA<Cat>::get_layoutA_TV() const {
  auto ref_A = make_layout(make_tuple(tile_size_mnk<0>(), tile_size_mnk<2>()));

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto atile = make_tile(
      _, make_tile(make_layout(make_tuple(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                               make_tuple(_1, _0)),
                   _));
  auto linear = make_layout(make_tuple(thr_layout_vmnk.size(), _1), make_tuple(_1, _0));
  auto thridx_2_thrid =
      composition(linear, right_inverse(make_layout(thr_layout_vmnk, complement(thr_layout_vmnk))));
  return thrfrg_A(ref_A).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class Cat> Layout<Cat> TiledMMA<Cat>::get_layoutB_TV() const {
  auto ref_B = make_layout(make_tuple(tile_size_mnk<1>(), tile_size_mnk<2>()));

  const auto &thr_layout_vmnk = get_thr_layout_vmnk();

  auto btile = make_tile(
      _, make_tile(make_layout(make_tuple(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                               make_tuple(_0, _1)),
                   _));
  auto linear = make_layout(make_tuple(thr_layout_vmnk.size(), _1), make_tuple(_1, _0));
  auto thridx_2_thrid =
      composition(linear, right_inverse(make_layout(thr_layout_vmnk, complement(thr_layout_vmnk))));
  return thrfrg_B(ref_B).compose(btile, _).compose(thridx_2_thrid, _);
}

template <class Cat>
template <class Engine>
auto ThrMMA<Cat>::partition_C(const Tensor<Cat, Engine> &ctensor) const {
  auto thr_tensor = this->thrfrg_C(ctensor);
  if (thr_tensor.isError())
    return thr_tensor;

  auto thr_vmn = make_tuple(get<0>(thr_vmnk_), make_tuple(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
  return thr_tensor(thr_vmn, make_tuple(_, repeat(_, rank<1, 1>(thr_tensor))));
}

template <class Cat>
template <class Engine>
auto ThrMMA<Cat>::partition_A(const Tensor<Cat, Engine> &atensor) const {
  auto thr_tensor = this->thrfrg_A(atensor);
  if (thr_tensor.isError())
    return thr_tensor;

  auto thr_vmk = make_tuple(get<0>(thr_vmnk_), make_tuple(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));
  return thr_tensor(thr_vmk, make_tuple(_, repeat(_, rank<1, 1>(thr_tensor))));
}

template <class Cat>
template <class Engine>
auto ThrMMA<Cat>::partition_B(const Tensor<Cat, Engine> &btensor) const {
  auto thr_tensor = this->thrfrg_B(btensor);
  if (thr_tensor.isError())
    return thr_tensor;

  auto thr_vnk = make_tuple(get<0>(thr_vmnk_), make_tuple(get<2>(thr_vmnk_), get<3>(thr_vmnk_)));
  return thr_tensor(thr_vnk, make_tuple(_, repeat(_, rank<1, 1>(thr_tensor))));
}

template <class Cat>
template <class ThrIdx, std::enable_if_t<is_int_tuple_v<ThrIdx>, int>>
auto TiledMMA<Cat>::get_slice(const ThrIdx &thr_idx) const {
  static_assert(std::is_same_v<Cat, category_of<ThrIdx>>);
  auto thr_vmnk = get_thr_layout_vmnk().get_flat_coord(thr_idx);
  return ThrMMA<Cat>(*this, std::move(thr_vmnk));
}

template <class Cat>
auto make_tiled_mma(const MMAAtom<Cat> &mma_atom, const Layout<Cat> &thr_layout,
                    const Tile<Cat> &permutations) {
  auto unit = make_layout(_1, _0);
  auto thrLayoutMNK = append(thr_layout, unit, 3);
  auto permutationMNK = append<3>(permutations, Tile<Cat>::getNone());
  return TiledMMA<Cat>(mma_atom, std::move(thrLayoutMNK), std::move(permutationMNK));
}

template <class Cat>
auto make_tiled_mma(const MMAAtom<Cat> &mma_atom, const Layout<Cat> &thr_layout) {
  return make_tiled_mma(mma_atom, thr_layout, Tile<Cat>());
}

template <class Cat> auto make_tiled_mma(const MMAAtom<Cat> &mma_atom) {
  auto thrLayout = make_layout(
      make_tuple(IntTuple<Cat>::getOne(), IntTuple<Cat>::getOne(), IntTuple<Cat>::getOne()));
  return make_tiled_mma(mma_atom, thrLayout, Tile<Cat>());
}

template <int I, class Cat> auto tile_size(const TiledMMA<Cat> &mma) {
  static_assert(I >= 0 && I < 3);
  return mma.template tile_size_mnk<I>();
}

template <class Cat> auto tile_shape(const TiledMMA<Cat> &mma) {
  return make_tuple(tile_size<0>(mma), tile_size<1>(mma), tile_size<2>(mma));
}

template <int... I, class Cat> auto size(const TiledMMA<Cat> &mma) {
  return size<I...>(mma.get_thr_layout_vmnk());
}

template <int... I, class Cat> auto thr_size(const TiledMMA<Cat> &mma) {
  return size<I...>(mma.get_thr_layout_vmnk());
}

template <class Cat, class Shape>
IntTuple<Cat> partition_shape_C(const TiledMMA<Cat> &mma, const Shape &shape_MN) {
  auto dummy = make_layout(shape(shape_MN));
  auto dummy_tv = mma.thrfrg_C(dummy);
  auto dummy_v = dummy_tv(_0, make_tuple(_, repeat(_, rank(dummy))));
  return dummy_v.shape();
}

template <class Cat, class Shape>
IntTuple<Cat> partition_shape_A(const TiledMMA<Cat> &mma, const Shape &shape_MK) {
  auto dummy = make_layout(shape(shape_MK));
  auto dummy_tv = mma.thrfrg_A(dummy);
  auto dummy_v = dummy_tv(_0, make_tuple(_, repeat(_, rank(dummy))));
  return dummy_v.shape();
}

template <class Cat, class Shape>
IntTuple<Cat> partition_shape_B(const TiledMMA<Cat> &mma, const Shape &shape_NK) {
  auto dummy = make_layout(shape(shape_NK));
  auto dummy_tv = mma.thrfrg_B(dummy);
  auto dummy_v = dummy_tv(_0, make_tuple(_, repeat(_, rank(dummy))));
  return dummy_v.shape();
}

template <class Cat> void print(const MMAAtom<Cat> &atom, llvm::raw_ostream &os) {
  os << "mma_atom<shape_mnk=";
  print(atom.shapeMNK(), os);
  os << ",thr_layout=";
  print(atom.thrLayout(), os);
  os << ",thr_val_a=";
  print(atom.thrValLayoutA(), os);
  os << ",thr_val_b=";
  print(atom.thrValLayoutB(), os);
  os << ",thr_val_c=";
  print(atom.thrValLayoutC(), os);
  os << '>';
}

template <class Cat> void print(const TiledMMA<Cat> &mma, llvm::raw_ostream &os) {
  os << "tiled_mma<";
  print(mma.mmaAtom(), os);
  os << ",atom_layout_mnk=";
  print(mma.atomLayoutMNK(), os);
  os << ",permutation_mnk=";
  print(mma.permutationMNK(), os);
  os << '>';
}

template <class Cat> void print(const ThrMMA<Cat> &mma, llvm::raw_ostream &os) {
  os << "thr_mma<";
  print(mma.tiledMMA(), os);
  os << ",thr_vmnk=";
  print(mma.threadCoord(), os);
  os << '>';
}

} // namespace mlir::fly::core

#include "flydsl/Core/Algebra/LiteralMacro.hpp.inc"

#endif // FLYDSL_CORE_ALGEBRA_TILED_MMA_HPP
