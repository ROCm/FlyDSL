# Registration entry points for every FlyDSL dialect and pass, shared by the
# tools and the C API. Each backend adds its hooks through its descriptor in
# cmake/backends/.
get_property(_backend_sources GLOBAL PROPERTY FLYDSL_BACKEND_REGISTRATION_SOURCES)
get_property(_backend_libs GLOBAL PROPERTY FLYDSL_BACKEND_REGISTRATION_LIBS)
add_mlir_library(MLIRFlyRegisterAll
  RegisterAll.cpp
  ${_backend_sources}

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyKtraceDialect
  ${_backend_libs}
)
