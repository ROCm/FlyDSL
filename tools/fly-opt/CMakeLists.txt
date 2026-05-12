get_property(_backend_opt_libs GLOBAL PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS)

set(LIBS
  MLIRFlyDialect
  ${_backend_opt_libs}
  MLIRRegisterAllExtensions

  MLIRMlirOptMain
  MLIRToLLVMIRTranslationRegistration
)

add_llvm_tool(fly-opt
  fly-opt.cpp
)

target_link_libraries(fly-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(fly-opt PRIVATE ${llvm_libs})
