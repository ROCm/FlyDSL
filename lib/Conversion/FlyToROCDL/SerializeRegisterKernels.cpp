// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_SERIALIZEREGISTERKERNELSPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

namespace mlir::fly {
void registerRegisterPlacementCodegen();
}

using namespace mlir;

namespace {
// An LLVM MC parser owns all instruction, encoding, and subtarget rules. Check
// both its result and MCContext errors; never infer success from output bytes.
FailureOr<SmallVector<char, 0>> assembleChecked(StringRef isa, ROCDL::ROCDLTargetAttr target,
                                                Operation &module) {
  llvm::Triple triple(llvm::Triple::normalize(target.getTriple()));
  std::string error;
  const llvm::Target *backend = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!backend)
    return module.emitError() << "LLVM assembler target lookup failed: " << error;
  llvm::MCTargetOptions options;
  std::unique_ptr<llvm::MCRegisterInfo> mri(backend->createMCRegInfo(triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(backend->createMCAsmInfo(*mri, triple, options));
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      backend->createMCSubtargetInfo(triple, target.getChip(), target.getFeatures()));
  llvm::SourceMgr source;
  source.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(isa), llvm::SMLoc());
  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  source.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *stream) {
        diag.print("FlyDSL LLVM assembler", *static_cast<llvm::raw_ostream *>(stream));
      },
      static_cast<llvm::raw_ostream *>(&diagnosticStream));
  llvm::MCContext context(triple, *mai, *mri, *sti, &source);
  context.setDiagnosticHandler([&](const llvm::SMDiagnostic &diag, bool, const llvm::SourceMgr &,
                                   std::vector<const llvm::MDNode *> &) {
    diag.print("FlyDSL LLVM assembler", diagnosticStream);
  });
  std::unique_ptr<llvm::MCObjectFileInfo> objectInfo(
      backend->createMCObjectFileInfo(context, /*PIC=*/false));
  context.setObjectFileInfo(objectInfo.get());
  SmallVector<char, 0> object;
  llvm::raw_svector_ostream stream(object);
  std::unique_ptr<llvm::MCInstrInfo> mii(backend->createMCInstrInfo());
  std::unique_ptr<llvm::MCCodeEmitter> emitter(backend->createMCCodeEmitter(*mii, context));
  std::unique_ptr<llvm::MCAsmBackend> asmBackend(backend->createMCAsmBackend(*sti, *mri, options));
  auto writer = asmBackend->createObjectWriter(stream);
  std::unique_ptr<llvm::MCStreamer> streamer(backend->createMCObjectStreamer(
      triple, context, std::move(asmBackend), std::move(writer), std::move(emitter), *sti));
  std::unique_ptr<llvm::MCAsmParser> parser(
      llvm::createMCAsmParser(source, context, *streamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> targetParser(
      backend->createMCAsmParser(*sti, *parser, *mii));
  if (!targetParser)
    return module.emitError("LLVM target assembler is unavailable");
  parser->setTargetParser(*targetParser);
  bool failed = parser->Run(/*NoInitialTextSection=*/false);
  if (failed || context.hadError())
    return module.emitError() << "explicit register kernel failed LLVM assembler validation:\n"
                              << diagnostics;
  return object;
}

// Reuse the upstream serializer's translation, linking, optimization, and
// target setup. Only the output boundary differs: assembly is checked before
// it can be returned or linked into a loadable kernel.
class CheckedSerializer : public ROCDL::SerializeGPUModuleBase {
  gpu::CompilationTarget format;

public:
  CheckedSerializer(gpu::GPUModuleOp module, ROCDL::ROCDLTargetAttr target,
                    const gpu::TargetOptions &options)
      : SerializeGPUModuleBase(*module, target, options), format(options.getCompilationTarget()) {}
  FailureOr<SmallVector<char, 0>> moduleToObject(llvm::Module &llvmModule) override {
    auto machine = getOrCreateTargetMachine();
    if (failed(machine))
      return failure();
    auto emitError = [&]() { return getOperation().emitError(); };
    auto isa = translateModuleToISA(llvmModule, **machine, emitError);
    if (failed(isa))
      return failure();
    auto object = assembleChecked(*isa, getTarget(), getOperation());
    if (failed(object))
      return failure();
    if (format == gpu::CompilationTarget::Assembly)
      return SmallVector<char, 0>(isa->begin(), isa->end());
    if (getToolkitPath().empty())
      return emitError() << "ROCm toolkit path is required to link register kernels";
    llvm::SmallString<128> linker(getToolkitPath());
    llvm::sys::path::append(linker, "llvm", "bin", "ld.lld");
    return ROCDL::linkObjectCode(*object, linker, emitError);
  }
};

bool hasRegisterPlacement(gpu::GPUModuleOp module) {
  for (auto func : module.getOps<LLVM::LLVMFuncOp>())
    if (auto attrs = func.getPassthroughAttr())
      for (Attribute attr : attrs)
        if (auto pair = dyn_cast<ArrayAttr>(attr); pair && pair.size() == 2)
          if (auto key = dyn_cast<StringAttr>(pair[0]); key && key == "flydsl-register-class")
            return true;
  return false;
}

class SerializeRegisterKernelsPass
    : public impl::SerializeRegisterKernelsPassBase<SerializeRegisterKernelsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    SmallVector<gpu::GPUModuleOp> modules;
    getOperation()->walk([&](gpu::GPUModuleOp module) {
      if (hasRegisterPlacement(module))
        modules.push_back(module);
    });
    if (modules.empty())
      return;
    fly::registerRegisterPlacementCodegen();
    auto output = llvm::StringSwitch<std::optional<gpu::CompilationTarget>>(format)
                      .Cases({"isa", "assembly"}, gpu::CompilationTarget::Assembly)
                      .Cases({"bin", "binary"}, gpu::CompilationTarget::Binary)
                      .Cases({"fatbin", "fatbinary"}, gpu::CompilationTarget::Fatbin)
                      .Default(std::nullopt);
    if (!output) {
      getOperation()->emitError(
          "explicit register serialization requires isa, bin, or fatbin format");
      return signalPassFailure();
    }
    ROCDL::SerializeGPUModuleBase::init();
    SmallVector<Attribute> libraries;
    for (const auto &path : linkFiles)
      libraries.push_back(StringAttr::get(&getContext(), path));
    for (auto module : modules) {
      if (!module.getTargetsAttr()) {
        module.emitError("explicit register kernel has no target");
        return signalPassFailure();
      }
      SymbolTable symbols(SymbolTable::getNearestSymbolTable(module->getParentOp()));
      auto getSymbols = [&]() { return &symbols; };
      gpu::TargetOptions options(toolkit, libraries, cmdOptions, section, *output, getSymbols);
      SmallVector<Attribute> objects;
      for (Attribute attr : module.getTargetsAttr()) {
        auto target = dyn_cast<ROCDL::ROCDLTargetAttr>(attr);
        if (!target) {
          module.emitError("explicit register serialization requires a ROCDL target");
          return signalPassFailure();
        }
        CheckedSerializer serializer(module, target, options);
        auto bytes = serializer.run();
        if (!bytes)
          return signalPassFailure();
        auto object = cast<gpu::TargetAttrInterface>(attr).createObject(
            module, gpu::SerializedObject{std::move(*bytes)}, options);
        if (!object) {
          module.emitError("failed to create explicit register kernel object");
          return signalPassFailure();
        }
        objects.push_back(object);
      }
      OpBuilder builder(module);
      gpu::BinaryOp::create(builder, module.getLoc(), module.getName(),
                            dyn_cast_or_null<gpu::OffloadingLLVMTranslationAttrInterface>(
                                module.getOffloadingHandlerAttr()),
                            builder.getArrayAttr(objects));
      module.erase();
    }
  }
};
} // namespace
