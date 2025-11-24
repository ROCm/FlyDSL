set(LIBS
  CuteDialect
  CuteTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
)

add_llvm_tool(cute-opt
  cute-opt.cpp
)

target_link_libraries(cute-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(cute-opt PRIVATE ${llvm_libs})
