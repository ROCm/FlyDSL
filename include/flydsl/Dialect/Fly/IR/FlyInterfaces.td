// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_INTERFACES 
#define FLY_INTERFACES

include "mlir/IR/OpBase.td"

def Fly_NestedInterfaceMethods {
  list<InterfaceMethod> methods = [
    InterfaceMethod<"", "bool", "isLeaf", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "::llvm::ArrayRef<int32_t>":$idxs)>,
    InterfaceMethod<"", "int32_t", "depth", (ins)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "::llvm::ArrayRef<int32_t>":$idxs)>
  ];
}

def Fly_MayStaticInterface {
  list<InterfaceMethod> methods = [
    InterfaceMethod<"", "bool", "isStatic", (ins)>
  ];
}

def Fly_StatefulOpTypeInterface : TypeInterface<"StatefulOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<
      "Return the converted type that holds the mutable state of this op.",
      "::mlir::Type", "getConvertedType", (ins "::mlir::MLIRContext *":$ctx)>,
    InterfaceMethod<
      "Build the default state value for this op.",
      "::mlir::Value", "getDefaultState",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc)>,
    InterfaceMethod<
      "Build the initial state value for an atom materialized with `args`.",
      "::mlir::Value", "getAtomState",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
           "::mlir::ValueRange":$args),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        if (!args.empty()) {
          ::mlir::emitError(loc) << "this atom takes no construction arguments, got "
                                 << args.size();
          return nullptr;
        }
        return $_type.getDefaultState(builder, loc);
      }]>,
    InterfaceMethod<
      "Insert the given value to a field of this stateful op. "
      "Returns the updated struct value after setting the field.",
      "::mlir::Value",
      "setAtomState",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Value":$atomStruct,
           "::mlir::Attribute":$fieldAttr,
           "::mlir::Value":$fieldValue)>
  ];
}

def Fly_NestedAttrInterface : AttrInterface<"NestedAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_NestedInterfaceMethods.methods;
}
def Fly_NestedTypeInterface : TypeInterface<"NestedTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_NestedInterfaceMethods.methods;
}

def Fly_MayStaticAttrInterface : AttrInterface<"MayStaticAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_MayStaticInterface.methods;
}
def Fly_MayStaticTypeInterface : TypeInterface<"MayStaticTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_MayStaticInterface.methods;
}

def Fly_GPUTargetInfoAttrInterface : AttrInterface<"GPUTargetInfoAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let description = [{
    Hardware facts of a `gpu.module` target that Fly passes depend on. A backend
    attaches it to its target attribute (e.g. `#rocdl.target`) as an external
    model, so Fly reads them from the target the module is compiled for without
    depending on the backend.
  }];
  let methods = [
    InterfaceMethod<
      "Return how much of `resource` one kernel may use, or std::nullopt when the target "
      "does not know the resource, which is then not checked.",
      "std::optional<int64_t>",
      "getResourceLimit",
      (ins "::mlir::Attribute":$resource)>
  ];
}

def Fly_KernelResourceAttrInterface : AttrInterface<"KernelResourceAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let description = [{
    Implemented by an attribute that names a hardware resource kernels use. As with
    address spaces, resources every target may have are Fly attributes (e.g.
    `#fly.kernel_resource<shared_memory>`), and a resource only some targets have is an
    attribute of the backend dialect that defines it. Target-neutral passes only compare
    and print resources; ops report how much they use through `ResourceUsageOpInterface`
    and targets report limits through `GPUTargetInfoAttrInterface`.
  }];
  let methods = [
    InterfaceMethod<"The unit an amount of the resource counts, e.g. \"bytes of shared memory\".",
      "::llvm::StringRef", "getDescription", (ins)>
  ];
}

def Fly_ResourceUsageOpInterface : OpInterface<"ResourceUsageOpInterface"> {
  let cppNamespace = "::mlir::fly";
  let description = [{
    Implemented by an op that allocates a statically known amount of a kernel resource,
    or frees such an allocation. An allocation is held from the op allocating it to the
    op freeing it, named by the same handle, or to the end of the kernel; a resource
    that is never freed, such as shared memory, has no op freeing it.
  }];
  let methods = [
    InterfaceMethod<"Append the allocations and frees this op performs to `usage`.",
      "void", "getResourceUsage",
      (ins "::llvm::SmallVectorImpl<::mlir::fly::ResourceUsage> &":$usage)>
  ];
}

def Fly_CopyOpTypeInterface : TypeInterface<"CopyOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrLayout", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutSrc", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutDst", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutRef", (ins)>,
    InterfaceMethod<
      "Emit the lowering IR for a copy_atom_call with this CopyOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcMemTy,
           "::mlir::Type":$dstMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst)>,
    InterfaceMethod<
      "Emit the lowering IR for a predicated copy_atom_call with this CopyOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcMemTy,
           "::mlir::Type":$dstMemTy,
           "::mlir::Type":$predMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst,
           "::mlir::Value":$pred)>,
    InterfaceMethod<
      "Emit SSA-form copy: reads src, returns vector result.",
      "::mlir::FailureOr<::mlir::Value>",
      "emitAtomCallSSA",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$resultTy,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcTy,
           "::mlir::Type":$dstTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst)>,
    InterfaceMethod<
      "Emit SSA-form predicated copy with a scalar i1 predicate (normalized by lowering); "
      "reads src under predicate and returns a result when resultTy is present.",
      "::mlir::FailureOr<::mlir::Value>",
      "emitAtomCallSSA",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$resultTy,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcTy,
           "::mlir::Type":$dstTy,
           "::mlir::Type":$predTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst,
           "::mlir::Value":$pred)>
  ];
}

def Fly_MmaOpTypeInterface : TypeInterface<"MmaOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrLayout", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getShapeMNK", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeA", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeB", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeC", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeD", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutA", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutB", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutC", (ins)>,
    InterfaceMethod<
      "Emit the lowering IR for an mma_atom_call with this MmaOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$mmaAtomTy,
           "::mlir::Type":$dMemTy,
           "::mlir::TypeRange":$aMemTy,
           "::mlir::TypeRange":$bMemTy,
           "::mlir::Type":$cMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$d,
           "::mlir::ValueRange":$a,
           "::mlir::ValueRange":$b,
           "::mlir::Value":$c)>,
    InterfaceMethod<"", "::mlir::FailureOr<::mlir::Value>", "emitAtomCallSSA",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$resultTy,
           "::mlir::Type":$mmaAtomTy,
           "::mlir::Type":$dTy,
           "::mlir::TypeRange":$aTy,
           "::mlir::TypeRange":$bTy,
           "::mlir::Type":$cTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$d,
           "::mlir::ValueRange":$a,
           "::mlir::ValueRange":$b,
           "::mlir::Value":$c)>,
    InterfaceMethod<
      "Whether every call of this MMA op modifies its C input, for example negates it. A GEMM "
      "passes a tile's partial sum as C after the tile's first K step, so it would modify that "
      "sum too.",
      "bool", "modifiesC", (ins)>
  ];
}

#endif // FLY_INTERFACES
