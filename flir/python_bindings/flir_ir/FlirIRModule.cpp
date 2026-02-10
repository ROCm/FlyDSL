//===- FlirIRModule.cpp - FLIR IR Python module ---------------------------===//
//
// Combines all MLIR Python bindings into a single module with hidden symbols.
// Based on MainModule.cpp from MLIR, with modifications for unified build
// and additional dialect/pass registrations.
//
//===----------------------------------------------------------------------===//

#include "Globals.h"
#include "IRModule.h"
#include "NanobindUtils.h"
#include "Pass.h"
#include "Rewrite.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

// MLIR C-API headers
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/CAPI/IR.h"

// FLIR dialect and passes
#include "flir/FlirDialect.h"
#include "flir/FlirPasses.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flir/FlirPasses.h.inc"
} // namespace mlir

namespace nb = nanobind;
using namespace mlir;
using namespace nb::literals;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

//===----------------------------------------------------------------------===//
// GPU Dialect Bindings (from DialectGPU.cpp)
//===----------------------------------------------------------------------===//

static void populateDialectGPUSubmodule(nb::module_ &m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__gpu__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
}

// Populate GPU types - must be called after IR module is populated
static void populateDialectGPUTypes(nb::module_ &gpuMod, nb::module_ &irMod) {
  nb::object typeClass = irMod.attr("Type");
  nb::object attrClass = irMod.attr("Attribute");

  // AsyncTokenType
  auto asyncTokenType =
      mlir_type_subclass(gpuMod, "AsyncTokenType", mlirTypeIsAGPUAsyncTokenType,
                         typeClass);
  asyncTokenType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirGPUAsyncTokenTypeGet(ctx));
      },
      "Gets an instance of AsyncTokenType", nb::arg("cls"),
      nb::arg("ctx").none() = nb::none());

  // ObjectAttr
  mlir_attribute_subclass(gpuMod, "ObjectAttr", mlirAttributeIsAGPUObjectAttr,
                          attrClass)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirAttribute target, uint32_t format,
             const nb::bytes &object,
             std::optional<MlirAttribute> mlirObjectProps,
             std::optional<MlirAttribute> mlirKernelsAttr) {
            MlirStringRef objectStrRef = mlirStringRefCreate(
                static_cast<char *>(const_cast<void *>(object.data())),
                object.size());
            return cls(mlirGPUObjectAttrGetWithKernels(
                mlirAttributeGetContext(target), target, format, objectStrRef,
                mlirObjectProps.has_value() ? *mlirObjectProps
                                            : MlirAttribute{nullptr},
                mlirKernelsAttr.has_value() ? *mlirKernelsAttr
                                            : MlirAttribute{nullptr}));
          },
          "cls"_a, "target"_a, "format"_a, "object"_a,
          "properties"_a.none() = nb::none(), "kernels"_a.none() = nb::none())
      .def_property_readonly(
          "target",
          [](MlirAttribute self) { return mlirGPUObjectAttrGetTarget(self); })
      .def_property_readonly(
          "format",
          [](MlirAttribute self) { return mlirGPUObjectAttrGetFormat(self); })
      .def_property_readonly("object",
                             [](MlirAttribute self) {
                               MlirStringRef stringRef =
                                   mlirGPUObjectAttrGetObject(self);
                               return nb::bytes(stringRef.data, stringRef.length);
                             })
      .def_property_readonly("properties",
                             [](MlirAttribute self) -> nb::object {
                               if (mlirGPUObjectAttrHasProperties(self))
                                 return nb::cast(
                                     mlirGPUObjectAttrGetProperties(self));
                               return nb::none();
                             })
      .def_property_readonly("kernels", [](MlirAttribute self) -> nb::object {
        if (mlirGPUObjectAttrHasKernels(self))
          return nb::cast(mlirGPUObjectAttrGetKernels(self));
        return nb::none();
      });
}

//===----------------------------------------------------------------------===//
// LLVM Dialect Bindings (from DialectLLVM.cpp)
//===----------------------------------------------------------------------===//

static void populateDialectLLVMSubmodule(nb::module_ &m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__llvm__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
}

//===----------------------------------------------------------------------===//
// ExecutionEngine Bindings (from ExecutionEngineModule.cpp)
//===----------------------------------------------------------------------===//

namespace {

class PyExecutionEngine {
public:
  PyExecutionEngine(MlirExecutionEngine executionEngine)
      : executionEngine(executionEngine) {}
  PyExecutionEngine(PyExecutionEngine &&other) noexcept
      : executionEngine(other.executionEngine) {
    other.executionEngine.ptr = nullptr;
  }
  ~PyExecutionEngine() {
    if (executionEngine.ptr != nullptr)
      mlirExecutionEngineDestroy(executionEngine);
  }
  MlirExecutionEngine get() { return executionEngine; }

private:
  MlirExecutionEngine executionEngine;
};

} // namespace

static void populateExecutionEngineSubmodule(nb::module_ &m) {
  nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
      .def(
          "__init__",
          [](PyExecutionEngine *self, PyModule &module, int optLevel,
             const std::vector<std::string> &sharedLibPaths,
             bool enableObjectDump) {
            llvm::SmallVector<MlirStringRef> libPaths;
            for (const std::string &path : sharedLibPaths)
              libPaths.push_back({path.c_str(), path.length()});
            MlirExecutionEngine engine = mlirExecutionEngineCreate(
                module.get(), optLevel, libPaths.size(), libPaths.data(),
                enableObjectDump);
            if (engine.ptr == nullptr)
              throw nb::value_error("Execution engine creation failed.");
            new (self) PyExecutionEngine(engine);
          },
          nb::arg("module"), nb::arg("opt_level") = 2,
          nb::arg("shared_libs") = nb::list(),
          nb::arg("enable_object_dump") = false)
      .def(
          "raw_lookup",
          [](PyExecutionEngine &self, const std::string &func) {
            auto *res = mlirExecutionEngineLookupPacked(
                self.get(), mlirStringRefCreate(func.c_str(), func.size()));
            return reinterpret_cast<uintptr_t>(res);
          },
          nb::arg("func_name"))
      .def(
          "raw_register_runtime",
          [](PyExecutionEngine &self, const std::string &name, nb::object func) {
            if (!nb::isinstance<nb::capsule>(func))
              throw nb::type_error("func is not a PyCapsule");
            nb::capsule capsule = nb::cast<nb::capsule>(func);
            void *ptr = capsule.data();
            mlirExecutionEngineRegisterSymbol(
                self.get(), mlirStringRefCreate(name.c_str(), name.size()), ptr);
          },
          nb::arg("name"), nb::arg("func"))
      .def("dump_to_object_file", [](PyExecutionEngine &self,
                                     const std::string &fileName) {
        mlirExecutionEngineDumpToObjectFile(
            self.get(), mlirStringRefCreate(fileName.c_str(), fileName.size()));
      })
      .def(
          "initialize",
          [](PyExecutionEngine &self) {
            mlirExecutionEngineInitialize(self.get());
          },
          "Initialize the ExecutionEngine. Runs global constructors.");
}

//===----------------------------------------------------------------------===//
// Module initialization
//===----------------------------------------------------------------------===//

NB_MODULE(_flir_ir, m) {
  m.doc() = "FLIR IR Python Extension";

  //--- _mlir submodule (core MLIR bindings) ---
  auto mlirMod = m.def_submodule("_mlir", "MLIR Python Bindings");

  nb::class_<PyGlobals>(mlirMod, "_Globals")
      .def_prop_rw("dialect_search_modules", &PyGlobals::getDialectSearchPrefixes,
                   &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix", &PyGlobals::addDialectSearchPrefix,
           "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a)
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, nb::kw_only(),
           "replace"_a = false)
      // Traceback-related methods (needed by generated dialect bindings)
      .def("loc_tracebacks_enabled",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebacksEnabled();
           })
      .def("set_loc_tracebacks_enabled",
           [](PyGlobals &self, bool enabled) {
             self.getTracebackLoc().setLocTracebacksEnabled(enabled);
           })
      .def("set_loc_tracebacks_frame_limit",
           [](PyGlobals &self, int n) {
             self.getTracebackLoc().setLocTracebackFramesLimit(n);
           })
      .def("register_traceback_file_inclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileInclusion(filename);
           })
      .def("register_traceback_file_exclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileExclusion(filename);
           });

  mlirMod.attr("globals") =
      nb::cast(new PyGlobals, nb::rv_policy::take_ownership);

  // Registration decorators
  mlirMod.def(
      "register_dialect",
      [](nb::type_object pyClass) {
        std::string dialectNamespace =
            nb::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a);

  mlirMod.def(
      "register_operation",
      [](const nb::type_object &dialectClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [dialectClass, replace](nb::type_object opClass) -> nb::type_object {
              std::string operationName =
                  nb::cast<std::string>(opClass.attr("OPERATION_NAME"));
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);
              nb::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      "dialect_class"_a, nb::kw_only(), "replace"_a = false);

  mlirMod.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [mlirTypeID, replace](nb::callable typeCaster) -> nb::object {
              PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster,
                                                  replace);
              return typeCaster;
            });
      },
      "typeid"_a, nb::kw_only(), "replace"_a = false);

  mlirMod.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [mlirTypeID, replace](nb::callable valueCaster) -> nb::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      "typeid"_a, nb::kw_only(), "replace"_a = false);

  // IR submodule
  auto irModule = mlirMod.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  // Rewrite submodule
  auto rewriteModule = mlirMod.def_submodule("rewrite", "MLIR Rewrite Bindings");
  populateRewriteSubmodule(rewriteModule);

  // PassManager submodule
  auto passModule =
      mlirMod.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);

  //--- Dialect submodules ---
  auto gpuDialectMod =
      m.def_submodule("_mlirDialectsGPU", "MLIR GPU Dialect Bindings");
  populateDialectGPUSubmodule(gpuDialectMod);
  populateDialectGPUTypes(gpuDialectMod, irModule);

  auto llvmDialectMod =
      m.def_submodule("_mlirDialectsLLVM", "MLIR LLVM Dialect Bindings");
  populateDialectLLVMSubmodule(llvmDialectMod);

  //--- GPU Passes submodule ---
  auto gpuPassesMod = m.def_submodule("_mlirGPUPasses", "MLIR GPU Passes");
  mlirRegisterGPUPasses();

  //--- ExecutionEngine submodule ---
  auto execEngineMod =
      m.def_submodule("_mlirExecutionEngine", "MLIR Execution Engine");
  populateExecutionEngineSubmodule(execEngineMod);

  //--- RegisterEverything submodule ---
  auto regEverythingMod = m.def_submodule("_mlirRegisterEverything",
                                          "MLIR Register All Dialects/Passes");
  regEverythingMod.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
    // Also register FLIR dialect
    auto *cppRegistry = unwrap(registry);
    cppRegistry->insert<mlir::flir::FlirDialect>();
  });
  regEverythingMod.def("register_llvm_translations", [](MlirContext context) {
    mlirRegisterAllLLVMTranslations(context);
  });
  mlirRegisterAllPasses();
  
  //--- FLIR Passes submodule (replaces _flirPasses.so) ---
  auto flirPassesMod = m.def_submodule("_flirPasses", "FLIR Dialect Passes");
  flirPassesMod.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
    auto *cppRegistry = unwrap(registry);
    cppRegistry->insert<mlir::flir::FlirDialect>();
  });
  flirPassesMod.def("register_llvm_translations", [](MlirContext context) {
    mlirRegisterAllLLVMTranslations(context);
  });
  // Register FLIR-specific passes
  ::mlir::registerFlirToStandardPass();
  ::mlir::registerFlirTrivialDCEPass();

  // NOTE: sys.modules aliases are registered in Python __init__.py, not here.
  // The Python side handles all three groups (C++ capsule, LLVM wrappers, local).
}
