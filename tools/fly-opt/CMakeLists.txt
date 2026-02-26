set(LIBS
  MLIRFlyDialect
  MLIRFlyROCDLDialect
  MLIRFlyToROCDL

  MLIRMlirOptMain
)

add_llvm_tool(fly-opt
  fly-opt.cpp
)

target_link_libraries(fly-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(fly-opt PRIVATE ${llvm_libs})
