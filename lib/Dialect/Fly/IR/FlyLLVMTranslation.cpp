//===- FlyLLVMTranslation.cpp - Fly LLVM translation helpers --------------===//
//
// Derived from LLVM Project: mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Explicit ROCm module loading for FlyDSL JIT.
//
// Keep the launch lowering in this file synchronized with LLVM MLIR's
// SelectObjectAttr implementation when updating the bundled MLIR version.
//
//===----------------------------------------------------------------------===//

#include "flydsl/Dialect/Fly/IR/FlyLLVMTranslation.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

static llvm::Twine getModuleIdentifier(llvm::StringRef moduleName) {
  return moduleName + "_module";
}

static std::string getAotStateIdentifier(llvm::StringRef moduleName) {
  return (moduleName + "_aot_state").str();
}

namespace {

gpu::ObjectAttr getFirstObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();
  if (objects.empty()) {
    op.emitError("expected at least one GPU object");
    return nullptr;
  }
  return dyn_cast<gpu::ObjectAttr>(objects.front());
}

LogicalResult embedExplicitModule(StringRef moduleName, gpu::ObjectAttr object,
                                  llvm::Module &module) {
  bool addNull = (object.getFormat() == gpu::CompilationTarget::Assembly);
  StringRef serializedStr = object.getObject().getValue();
  llvm::Constant *serializedCst =
      llvm::ConstantDataArray::getString(module.getContext(), serializedStr, addNull);
  auto *serializedObj = new llvm::GlobalVariable(module, serializedCst->getType(), true,
                                                 llvm::GlobalValue::InternalLinkage, serializedCst,
                                                 moduleName + "_binary");
  serializedObj->setAlignment(llvm::MaybeAlign(8));
  serializedObj->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);

  auto optLevel = llvm::APInt::getZero(32);
  if (DictionaryAttr objectProps = object.getProperties()) {
    if (auto section = dyn_cast_or_null<StringAttr>(objectProps.get(gpu::elfSectionName))) {
      serializedObj->setSection(section.getValue());
    }
    if (auto optAttr = dyn_cast_or_null<IntegerAttr>(objectProps.get("O")))
      optLevel = optAttr.getValue();
  }

  llvm::IRBuilder<> builder(module.getContext());
  auto *i32Ty = builder.getInt32Ty();
  auto *i64Ty = builder.getInt64Ty();
  auto *ptrTy = builder.getPtrTy(0);
  auto *voidTy = builder.getVoidTy();

  auto *modulePtr = new llvm::GlobalVariable(
      module, ptrTy, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantPointerNull::get(ptrTy), getModuleIdentifier(moduleName));

  // init/load take their outputs as explicit pointer args
  // (void** module_out, int32_t* err) so they fit MLIR's packed-args wrapper.
  auto *initFn =
      llvm::Function::Create(llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, /*isVarArg=*/false),
                             llvm::GlobalValue::ExternalLinkage, "flydsl_gpu_module_init", module);
  auto *initBlock = llvm::BasicBlock::Create(module.getContext(), "entry", initFn);
  builder.SetInsertPoint(initBlock);
  auto initArgs = initFn->arg_begin();
  llvm::Value *outModulePtr = initArgs++;
  llvm::Value *errPtr = initArgs++;
  builder.CreateStore(llvm::ConstantPointerNull::get(ptrTy), outModulePtr);
  builder.CreateStore(llvm::ConstantInt::get(i32Ty, 0), errPtr);
  builder.CreateRetVoid();

  auto *loadFn = llvm::Function::Create(
      llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "flydsl_gpu_module_load_to_device", module);
  auto *loadBlock = llvm::BasicBlock::Create(module.getContext(), "entry", loadFn);
  builder.SetInsertPoint(loadBlock);
  auto loadArgs = loadFn->arg_begin();
  llvm::Value *moduleOutPtr = loadArgs++;
  llvm::Value *loadErrPtr = loadArgs++;
  llvm::Value *moduleObj = nullptr;
  if (object.getFormat() == gpu::CompilationTarget::Assembly) {
    llvm::FunctionCallee moduleLoadFn = module.getOrInsertFunction(
        "mgpuModuleLoadJIT", llvm::FunctionType::get(ptrTy, {ptrTy, i32Ty}, false));
    llvm::Constant *optValue = llvm::ConstantInt::get(i32Ty, optLevel);
    moduleObj = builder.CreateCall(moduleLoadFn, {serializedObj, optValue});
  } else {
    llvm::FunctionCallee moduleLoadFn = module.getOrInsertFunction(
        "mgpuModuleLoad", llvm::FunctionType::get(ptrTy, {ptrTy, i64Ty}, false));
    llvm::Constant *binarySize =
        llvm::ConstantInt::get(i64Ty, serializedStr.size() + (addNull ? 1 : 0));
    moduleObj = builder.CreateCall(moduleLoadFn, {serializedObj, binarySize});
  }
  builder.CreateStore(moduleObj, modulePtr);
  builder.CreateStore(moduleObj, moduleOutPtr);
  llvm::Value *isNull = builder.CreateICmpEQ(moduleObj, llvm::ConstantPointerNull::get(ptrTy));
  llvm::Value *errValue = builder.CreateSelect(isNull, llvm::ConstantInt::getSigned(i32Ty, -1),
                                               llvm::ConstantInt::get(i32Ty, 0));
  builder.CreateStore(errValue, loadErrPtr);
  builder.CreateRetVoid();

  return success();
}

// Embed `object` for an exported host object.  The binary and the module state
// slot stay internal; only the prefixed lifecycle functions are external.
LogicalResult embedAotModule(gpu::BinaryOp op, gpu::ObjectAttr object, StringRef symbolPrefix,
                             llvm::Module &module) {
  if (object.getFormat() == gpu::CompilationTarget::Assembly)
    return op.emitError("fly.aot_module requires a binary GPU object, not assembly");
  if (symbolPrefix.empty())
    return op.emitError("fly.aot_module requires a non-empty symbol prefix");
  StringRef moduleName = op.getName();

  StringRef serializedStr = object.getObject().getValue();
  llvm::Constant *serializedCst =
      llvm::ConstantDataArray::getString(module.getContext(), serializedStr, /*AddNull=*/false);
  auto *serializedObj = new llvm::GlobalVariable(module, serializedCst->getType(), true,
                                                 llvm::GlobalValue::InternalLinkage, serializedCst,
                                                 moduleName + "_binary");
  serializedObj->setAlignment(llvm::MaybeAlign(8));
  if (DictionaryAttr objectProps = object.getProperties()) {
    if (auto section = dyn_cast_or_null<StringAttr>(objectProps.get(gpu::elfSectionName)))
      serializedObj->setSection(section.getValue());
  }

  llvm::IRBuilder<> builder(module.getContext());
  auto *i32Ty = builder.getInt32Ty();
  auto *ptrTy = builder.getPtrTy(0);

  auto *statePtr = new llvm::GlobalVariable(
      module, ptrTy, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantPointerNull::get(ptrTy), getAotStateIdentifier(moduleName));

  // Emit `int32_t <prefix>__<suffix>(params...)` forwarding to
  // `runtimeName(leadingArgs..., params...)`.
  auto createLifecycleFn = [&](StringRef suffix, ArrayRef<llvm::Type *> params,
                               StringRef runtimeName,
                               ArrayRef<llvm::Value *> leadingArgs) -> llvm::Function * {
    std::string name = (symbolPrefix + "__" + suffix).str();
    if (module.getFunction(name))
      return nullptr;
    auto *fn = llvm::Function::Create(llvm::FunctionType::get(i32Ty, params, false),
                                      llvm::GlobalValue::ExternalLinkage, name, module);
    builder.SetInsertPoint(llvm::BasicBlock::Create(module.getContext(), "entry", fn));
    SmallVector<llvm::Value *> args(leadingArgs);
    for (llvm::Argument &arg : fn->args())
      args.push_back(&arg);
    SmallVector<llvm::Type *> argTypes;
    for (llvm::Value *arg : args)
      argTypes.push_back(arg->getType());
    llvm::FunctionCallee runtimeFn =
        module.getOrInsertFunction(runtimeName, llvm::FunctionType::get(i32Ty, argTypes, false));
    builder.CreateRet(builder.CreateCall(runtimeFn, args));
    return fn;
  };

  if (!createLifecycleFn("module_init", {}, "flydslAotModuleInit", {statePtr, serializedObj}) ||
      !createLifecycleFn("module_load", {i32Ty}, "flydslAotModuleLoad", {statePtr}) ||
      !createLifecycleFn("module_unload", {}, "flydslAotModuleUnload", {statePtr}))
    return op.emitError("fly.aot_module supports one GPU binary per symbol prefix; '")
           << symbolPrefix << "' is already used";
  return success();
}

} // namespace

namespace llvm {

class LaunchKernel {
public:
  LaunchKernel(Module &module, IRBuilderBase &builder,
               mlir::LLVM::ModuleTranslation &moduleTranslation)
      : module(module), builder(builder), moduleTranslation(moduleTranslation) {
    i32Ty = builder.getInt32Ty();
    i64Ty = builder.getInt64Ty();
    ptrTy = builder.getPtrTy(0);
    voidTy = builder.getVoidTy();
    intPtrTy = builder.getIntPtrTy(module.getDataLayout());
  }

  FunctionCallee getKernelLaunchFn() {
    return module.getOrInsertFunction(
        "mgpuLaunchKernel",
        FunctionType::get(voidTy,
                          ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                                            intPtrTy, i32Ty, ptrTy, ptrTy, ptrTy, i64Ty}),
                          false));
  }

  FunctionCallee getClusterKernelLaunchFn() {
    return module.getOrInsertFunction(
        "mgpuLaunchClusterKernel",
        FunctionType::get(
            voidTy,
            ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                              intPtrTy, intPtrTy, intPtrTy, i32Ty, ptrTy, ptrTy, ptrTy, i64Ty}),
            false));
  }

  FunctionCallee getModuleFunctionFn() {
    return module.getOrInsertFunction(
        "mgpuModuleGetFunction", FunctionType::get(ptrTy, ArrayRef<Type *>({ptrTy, ptrTy}), false));
  }

  FunctionCallee getAotKernelLaunchFn() {
    return module.getOrInsertFunction(
        "flydslAotModuleLaunchKernel",
        FunctionType::get(voidTy,
                          ArrayRef<Type *>({ptrTy, ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                                            intPtrTy, intPtrTy, i32Ty, ptrTy, ptrTy, ptrTy, i64Ty}),
                          false));
  }

  FunctionCallee getAotClusterKernelLaunchFn() {
    return module.getOrInsertFunction(
        "flydslAotModuleLaunchClusterKernel",
        FunctionType::get(voidTy,
                          ArrayRef<Type *>({ptrTy, ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                                            intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, i32Ty,
                                            ptrTy, ptrTy, ptrTy, i64Ty}),
                          false));
  }

  FunctionCallee getStreamCreateFn() {
    return module.getOrInsertFunction("mgpuStreamCreate", FunctionType::get(ptrTy, false));
  }

  FunctionCallee getStreamDestroyFn() {
    return module.getOrInsertFunction("mgpuStreamDestroy",
                                      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
  }

  FunctionCallee getStreamSyncFn() {
    return module.getOrInsertFunction("mgpuStreamSynchronize",
                                      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
  }

  Value *getOrCreateFunctionName(StringRef moduleName, StringRef kernelName) {
    std::string globalName = std::string(formatv("{0}_{1}_name", moduleName, kernelName));
    if (GlobalVariable *gv = module.getGlobalVariable(globalName, true))
      return gv;
    return builder.CreateGlobalString(kernelName, globalName);
  }

  Value *createKernelArgArray(mlir::gpu::LaunchFuncOp op) {
    SmallVector<Value *> args = moduleTranslation.lookupValues(op.getKernelOperands());
    SmallVector<Type *> structTypes(args.size(), nullptr);

    for (auto [i, arg] : llvm::enumerate(args))
      structTypes[i] = arg->getType();

    Type *structTy = StructType::create(module.getContext(), structTypes);
    Value *argStruct = builder.CreateAlloca(structTy, 0u);
    Value *argArray = builder.CreateAlloca(ptrTy, ConstantInt::get(intPtrTy, structTypes.size()));

    for (auto [i, arg] : enumerate(args)) {
      Value *structMember = builder.CreateStructGEP(structTy, argStruct, i);
      builder.CreateStore(arg, structMember);
      Value *arrayMember = builder.CreateConstGEP1_32(ptrTy, argArray, i);
      builder.CreateStore(structMember, arrayMember);
    }
    return argArray;
  }

  llvm::LogicalResult createKernelLaunch(mlir::gpu::LaunchFuncOp op, bool aotModule = false) {
    auto llvmValue = [&](mlir::Value value) -> Value * {
      Value *v = moduleTranslation.lookupValue(value);
      assert(v && "Value has not been translated.");
      return v;
    };

    mlir::gpu::KernelDim3 grid = op.getGridSizeOperandValues();
    Value *gx = llvmValue(grid.x), *gy = llvmValue(grid.y), *gz = llvmValue(grid.z);

    mlir::gpu::KernelDim3 block = op.getBlockSizeOperandValues();
    Value *bx = llvmValue(block.x), *by = llvmValue(block.y), *bz = llvmValue(block.z);

    Value *dynamicMemorySize = nullptr;
    if (mlir::Value dynSz = op.getDynamicSharedMemorySize())
      dynamicMemorySize = llvmValue(dynSz);
    else
      dynamicMemorySize = ConstantInt::get(i32Ty, 0);

    Value *argArray = createKernelArgArray(op);

    StringRef moduleName = op.getKernelModuleName().getValue();
    std::string moduleIdentifier =
        aotModule ? getAotStateIdentifier(moduleName) : getModuleIdentifier(moduleName).str();
    Value *modulePtr = module.getGlobalVariable(moduleIdentifier, true);
    if (!modulePtr)
      return op.emitError() << "Couldn't find the binary: " << moduleIdentifier;
    Value *functionName = getOrCreateFunctionName(moduleName, op.getKernelName());
    Value *moduleFunction = nullptr;
    if (!aotModule) {
      Value *moduleObj = builder.CreateLoad(ptrTy, modulePtr);
      moduleFunction = builder.CreateCall(getModuleFunctionFn(), {moduleObj, functionName});
    }

    Value *stream = nullptr;
    if (mlir::Value asyncObject = op.getAsyncObject()) {
      stream = llvmValue(asyncObject);
    } else if (!op.getAsyncDependencies().empty()) {
      // FlyDSL carries the HIP stream as the first gpu.launch_func async
      // dependency.  This is a FlyDSL convention, not generic MLIR
      // gpu.async.token semantics.
      stream = llvmValue(op.getAsyncDependencies().front());
    } else {
      stream = ConstantPointerNull::get(ptrTy);
    }

    llvm::Constant *paramsCount = llvm::ConstantInt::get(i64Ty, op.getNumKernelOperands());
    Value *nullPtr = ConstantPointerNull::get(ptrTy);

    if (op.hasClusterSize()) {
      mlir::gpu::KernelDim3 cluster = op.getClusterSizeOperandValues();
      Value *cx = llvmValue(cluster.x), *cy = llvmValue(cluster.y), *cz = llvmValue(cluster.z);
      if (aotModule) {
        builder.CreateCall(
            getAotClusterKernelLaunchFn(),
            ArrayRef<Value *>({modulePtr, functionName, cx, cy, cz, gx, gy, gz, bx, by, bz,
                               dynamicMemorySize, stream, argArray, nullPtr, paramsCount}));
      } else {
        builder.CreateCall(
            getClusterKernelLaunchFn(),
            ArrayRef<Value *>({moduleFunction, cx, cy, cz, gx, gy, gz, bx, by, bz,
                               dynamicMemorySize, stream, argArray, nullPtr, paramsCount}));
      }
    } else {
      if (aotModule) {
        builder.CreateCall(
            getAotKernelLaunchFn(),
            ArrayRef<Value *>({modulePtr, functionName, gx, gy, gz, bx, by, bz, dynamicMemorySize,
                               stream, argArray, nullPtr, paramsCount}));
      } else {
        builder.CreateCall(
            getKernelLaunchFn(),
            ArrayRef<Value *>({moduleFunction, gx, gy, gz, bx, by, bz, dynamicMemorySize, stream,
                               argArray, nullPtr, paramsCount}));
      }
    }

    return success();
  }

private:
  Module &module;
  IRBuilderBase &builder;
  mlir::LLVM::ModuleTranslation &moduleTranslation;
  Type *i32Ty{};
  Type *i64Ty{};
  Type *voidTy{};
  Type *intPtrTy{};
  PointerType *ptrTy{};
};

} // namespace llvm

namespace {

class ExplicitModuleAttrImpl
    : public gpu::OffloadingLLVMTranslationAttrInterface::FallbackModel<ExplicitModuleAttrImpl> {
public:
  LogicalResult embedBinary(Attribute attribute, Operation *operation, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) const {
    auto op = dyn_cast_or_null<gpu::BinaryOp>(operation);
    if (!op)
      return operation->emitError("operation must be a GPU binary"), failure();
    gpu::ObjectAttr object = getFirstObject(op);
    if (!object)
      return failure();
    return embedExplicitModule(op.getName(), object, *moduleTranslation.getLLVMModule());
  }

  LogicalResult launchKernel(Attribute attribute, Operation *launchFuncOp, Operation *binaryOp,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) const {
    auto op = dyn_cast_or_null<gpu::LaunchFuncOp>(launchFuncOp);
    if (!op)
      return launchFuncOp->emitError("operation must be a GPU launch func Op."), failure();
    if (!isa_and_nonnull<gpu::BinaryOp>(binaryOp))
      return binaryOp->emitError("operation must be a GPU binary."), failure();
    return llvm::LaunchKernel(*moduleTranslation.getLLVMModule(), builder, moduleTranslation)
        .createKernelLaunch(op);
  }
};

class AotModuleAttrImpl
    : public gpu::OffloadingLLVMTranslationAttrInterface::FallbackModel<AotModuleAttrImpl> {
public:
  LogicalResult embedBinary(Attribute attribute, Operation *operation, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) const {
    auto op = dyn_cast_or_null<gpu::BinaryOp>(operation);
    if (!op)
      return operation->emitError("operation must be a GPU binary"), failure();
    gpu::ObjectAttr object = getFirstObject(op);
    if (!object)
      return failure();
    return embedAotModule(op, object, cast<fly::AotModuleAttr>(attribute).getSymbolPrefix(),
                          *moduleTranslation.getLLVMModule());
  }

  LogicalResult launchKernel(Attribute attribute, Operation *launchFuncOp, Operation *binaryOp,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) const {
    auto op = dyn_cast_or_null<gpu::LaunchFuncOp>(launchFuncOp);
    if (!op)
      return launchFuncOp->emitError("operation must be a GPU launch func Op."), failure();
    if (!isa_and_nonnull<gpu::BinaryOp>(binaryOp))
      return binaryOp->emitError("operation must be a GPU binary."), failure();
    return llvm::LaunchKernel(*moduleTranslation.getLLVMModule(), builder, moduleTranslation)
        .createKernelLaunch(op, /*aotModule=*/true);
  }
};

} // namespace

void mlir::fly::registerExplicitModuleOffloadingLLVMTranslation(MLIRContext &context) {
  ExplicitModuleAttr::attachInterface<ExplicitModuleAttrImpl>(context);
  AotModuleAttr::attachInterface<AotModuleAttrImpl>(context);
}
