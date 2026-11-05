# The Runtime and Symbol categories evaluate Core concretely and symbolically
# for the unit tests. The Fly dialect never uses them, so only
# tests/core/CMakeLists.txt adds this directory.
add_mlir_library(FlyCoreTesting
  Runtime.cpp
  Symbol.cpp

  DISABLE_INSTALL
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  FlyCore
)
