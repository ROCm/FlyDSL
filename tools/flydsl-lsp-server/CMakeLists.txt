set(LIBS
  MLIRFlyRegisterAll
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)

add_llvm_tool(flydsl-lsp-server
  flydsl-lsp-server.cpp
)

target_link_libraries(flydsl-lsp-server PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(flydsl-lsp-server PRIVATE ${llvm_libs})
