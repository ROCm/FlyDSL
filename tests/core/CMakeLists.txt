if(NOT TARGET llvm_gtest)
  message(FATAL_ERROR
    "llvm_gtest is required for the Core tests. Point MLIR_DIR at an LLVM build "
    "tree, or at an install configured with -DLLVM_INSTALL_GTEST=ON "
    "(scripts/build_llvm.sh does this).")
endif()

# Core tests compile large header-only libraries.  Use lighter flags than the
# LLVM defaults.
function(fly_apply_fast_test_compile_options target)
  target_compile_options(${target} PRIVATE
    -std=c++17
    -fno-exceptions
    -fno-rtti
    -O0
    -g1
    -Wall
    -Wextra
    -Wno-unused-parameter
  )
  target_compile_definitions(${target} PRIVATE
    GTEST_NO_LLVM_SUPPORT=0
    __STDC_CONSTANT_MACROS
    __STDC_FORMAT_MACROS
    __STDC_LIMIT_MACROS
  )
endfunction()

# Compile each suite once as an OBJECT library.  Suite binaries and the
# combined flydsl-core-tests executable then link those objects; a source change
# rebuilds only that suite's objects, and ctest can still run suites in
# parallel.  llvm_gtest_main is linked only into executables so a single
# combined binary does not get duplicate mains.
#
#   add_core_test(<name> <source> [LINK_LIBS <lib>...])
function(add_core_test name source)
  cmake_parse_arguments(ARG "" "" "LINK_LIBS" ${ARGN})
  add_library(${name}Obj OBJECT ${source})
  target_include_directories(${name}Obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name}Obj PRIVATE FlyCoreTesting llvm_gtest ${ARG_LINK_LIBS})
  fly_apply_fast_test_compile_options(${name}Obj)

  add_executable(${name} $<TARGET_OBJECTS:${name}Obj>)
  target_link_libraries(${name} PRIVATE
    FlyCoreTesting
    ${ARG_LINK_LIBS}
    llvm_gtest
    llvm_gtest_main
  )
  add_test(NAME ${name} COMMAND ${name})
  set_property(GLOBAL APPEND PROPERTY FLYDSL_CORE_TEST_TARGETS ${name})
  set_property(GLOBAL APPEND PROPERTY FLYDSL_CORE_TEST_OBJECTS ${name}Obj)
  set_property(GLOBAL APPEND PROPERTY FLYDSL_CORE_TEST_LIBS ${ARG_LINK_LIBS})
endfunction()

# The test-only Runtime and Symbol categories (FlyCoreTesting).
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/Core/Testing ${CMAKE_CURRENT_BINARY_DIR}/CoreTesting)

# The algebra of each Core/Algebra class, checked against a reference.
foreach(class IN ITEMS Leaf IntTuple Layout Tile Swizzle ComposedLayout Pointer Tensor TiledCopy
                       TiledMMA)
  add_core_test(Core${class}ParityTests ${class}Test.cpp)
endforeach()

# What a category adds to the shared algebra, which no reference shows.
foreach(category IN ITEMS Trait Symbol)
  add_core_test(Core${category}Tests ${category}Test.cpp)
endforeach()

# The EmitOp category: its leaf operations lowered to LLVM and executed against
# Runtime, and its bindings to SSA values.
add_core_test(CoreEmitOpTests EmitOpTest.cpp LINK_LIBS
  MLIRFuncDialect
  MLIRExecutionEngine MLIRArithToLLVM MLIRControlFlowToLLVM MLIRSCFToControlFlow
  MLIRFuncToLLVM MLIRReconcileUnrealizedCasts
  MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
)

# Every suite in one binary; CI builds and runs this one.
get_property(FLYDSL_CORE_TEST_OBJECTS GLOBAL PROPERTY FLYDSL_CORE_TEST_OBJECTS)
get_property(FLYDSL_CORE_TEST_LIBS GLOBAL PROPERTY FLYDSL_CORE_TEST_LIBS)
set(_fly_core_test_objects)
foreach(_obj IN LISTS FLYDSL_CORE_TEST_OBJECTS)
  list(APPEND _fly_core_test_objects $<TARGET_OBJECTS:${_obj}>)
endforeach()

add_executable(flydsl-core-tests ${_fly_core_test_objects})
target_link_libraries(flydsl-core-tests PRIVATE
  FlyCoreTesting
  ${FLYDSL_CORE_TEST_LIBS}
  llvm_gtest
  llvm_gtest_main
)

get_property(FLYDSL_CORE_TEST_TARGETS GLOBAL PROPERTY FLYDSL_CORE_TEST_TARGETS)
add_custom_target(check-flydsl-core
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS ${FLYDSL_CORE_TEST_TARGETS}
  USES_TERMINAL
  COMMENT "Building and running all Core tests"
)
