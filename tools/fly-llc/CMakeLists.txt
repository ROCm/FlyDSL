# fly-llc: IR -> object emitter with injectable MIR passes (codegen plugin path).
add_llvm_tool(fly-llc
  fly-llc.cpp
  EXPORT_SYMBOLS  # export symbols so dlopen'd MIR pass plugins resolve LLVM symbols
)

llvm_map_components_to_libnames(fly_llc_libs
  AllTargetsCodeGens
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  CodeGen
  Core
  IRReader
  MC
  Passes
  Support
  Target
  TargetParser
)
target_link_libraries(fly-llc PRIVATE ${fly_llc_libs} ${CMAKE_DL_LIBS})
