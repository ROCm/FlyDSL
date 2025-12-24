add_mlir_public_c_api_library(MLIRCAPIFlir
  FlirDialect.cpp
  LINK_LIBS PUBLIC
  FlirDialect
  FlirTransforms
)

