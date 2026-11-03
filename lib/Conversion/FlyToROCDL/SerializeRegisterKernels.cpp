// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "RegisterPlacementDiagnostic.h"
#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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
#define GEN_PASS_DEF_ROCDLSERIALIZEREGISTERKERNELSPASS
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

// Collect LLVM error diagnostics. Without a handler LLVM only prints them and
// codegen still succeeds, so they would never reach Python.
struct CollectedDiagnostics : llvm::DiagnosticHandler {
  SmallVector<std::pair<std::optional<uint64_t>, std::string>> errors;

  bool handleDiagnostics(const llvm::DiagnosticInfo &info) override {
    if (info.getSeverity() != llvm::DS_Error)
      return false;
    std::string text;
    llvm::raw_string_ostream stream(text);
    llvm::DiagnosticPrinterRawOStream printer(stream);
    info.print(printer);
    std::optional<uint64_t> request;
    if (auto *placement = dyn_cast<fly::RegisterPlacementDiagnostic>(&info))
      request = placement->getRequest();
    errors.emplace_back(request, std::move(text));
    return true;
  }
};

// Reuse the upstream serializer's translation, linking, optimization, and
// target setup. Only the output boundary differs: placement diagnostics become
// MLIR errors at the requesting operation, and assembly is checked before it
// can be returned or linked into a loadable kernel.
class CheckedSerializer : public ROCDL::SerializeGPUModuleBase {
  gpu::CompilationTarget format;
  const DenseMap<uint64_t, Location> &requests;

public:
  CheckedSerializer(gpu::GPUModuleOp module, ROCDL::ROCDLTargetAttr target,
                    const gpu::TargetOptions &options, const DenseMap<uint64_t, Location> &requests)
      : SerializeGPUModuleBase(*module, target, options), format(options.getCompilationTarget()),
        requests(requests) {}

  FailureOr<SmallVector<char, 0>> moduleToObject(llvm::Module &llvmModule) override {
    auto machine = getOrCreateTargetMachine();
    if (failed(machine))
      return failure();
    auto emitError = [&]() { return getOperation().emitError(); };
    // Rejected requests compile without placement and are reported here,
    // instead of aborting the process.
    llvmModule.addModuleFlag(llvm::Module::Warning, fly::RegisterDiagnosticsFlag, 1);
    auto handler = std::make_unique<CollectedDiagnostics>();
    auto *collected = handler.get();
    llvmModule.getContext().setDiagnosticHandler(std::move(handler));

    auto isa = translateModuleToISA(llvmModule, **machine, emitError);
    for (const auto &[request, text] : collected->errors) {
      Location loc = getOperation().getLoc();
      if (request)
        if (auto it = requests.find(*request); it != requests.end())
          loc = it->second;
      mlir::emitError(loc) << text;
    }
    if (!collected->errors.empty() || failed(isa))
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

class ROCDLSerializeRegisterKernelsPass
    : public impl::ROCDLSerializeRegisterKernelsPassBase<ROCDLSerializeRegisterKernelsPass> {
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
      // Map the request ID of each lowered marker to the DSL operation.
      DenseMap<uint64_t, Location> requests;
      module.walk([&](LLVM::CallOp call) {
        auto callee = call.getCallee();
        if (!callee || !callee->starts_with("__flydsl_register_value_"))
          return;
        auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(*callee);
        if (!func || !func.getPassthroughAttr())
          return;
        for (Attribute attr : func.getPassthroughAttr()) {
          auto pair = dyn_cast<ArrayAttr>(attr);
          if (!pair || pair.size() != 2 ||
              pair[0] != StringAttr::get(&getContext(), "flydsl-register-request"))
            continue;
          uint64_t id;
          if (auto value = dyn_cast<StringAttr>(pair[1]);
              value && !value.getValue().getAsInteger(10, id))
            requests.try_emplace(id, call.getLoc());
        }
      });

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
        CheckedSerializer serializer(module, target, options, requests);
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
