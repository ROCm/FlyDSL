set(LIBS
  MLIRFlyDialect
  MLIRRegisterAllExtensions

  MLIRMlirOptMain
)

if(FLYDSL_BUILD_ROCDL_STACK)
  list(APPEND LIBS
    MLIRFlyROCDLDialect
    MLIRFlyToROCDL
  )
endif()

add_llvm_tool(fly-opt
  fly-opt.cpp
)

target_link_libraries(fly-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(fly-opt PRIVATE ${llvm_libs})

if(FLYDSL_BUILD_ROCDL_STACK)
  target_compile_definitions(fly-opt PRIVATE FLYDSL_HAS_ROCDL_TARGET_STACK)
endif()
