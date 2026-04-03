# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Active Fly target lowering stack (one per build). Sets FLYDSL_TARGET_STACK and
# stack-specific variables for other CMakeLists.

set(FLYDSL_TARGET_STACK "rocdl"
    CACHE STRING "Active Fly target lowering stack for this build (one stack per artifact).")
set_property(CACHE FLYDSL_TARGET_STACK PROPERTY STRINGS rocdl)

# Extend the list and the if-branch when supporting additional target stacks.
set(_FLYDSL_TARGET_STACK_ALLOWED rocdl)
if(NOT FLYDSL_TARGET_STACK IN_LIST _FLYDSL_TARGET_STACK_ALLOWED)
  message(FATAL_ERROR
    "FLYDSL_TARGET_STACK='${FLYDSL_TARGET_STACK}' is not supported. "
    "Allowed values: ${_FLYDSL_TARGET_STACK_ALLOWED}")
endif()

set(FLYDSL_BUILD_ROCDL_STACK OFF)
if(FLYDSL_TARGET_STACK STREQUAL "rocdl")
  set(FLYDSL_BUILD_ROCDL_STACK ON)
  set(FLYDSL_STACK_MLIR_TARGET_DIALECT_NAME "fly_rocdl")
  set(FLYDSL_STACK_CPP_TARGET_TOKEN "ROCDL")
  set(FLYDSL_STACK_UPSTREAM_MLIR_PY_DIALECT "rocdl")

  set(FLYDSL_REGISTER_EXTRA_INCLUDES "#include \"flydsl-c/FlyROCDLDialect.h\"")
  set(FLYDSL_REGISTER_TARGET_DIALECT_HANDLES [[    MlirDialectHandle flyROCDLHandle = mlirGetDialectHandle__fly_rocdl__();
    mlirDialectHandleInsertDialect(flyROCDLHandle, registry);]])
  set(FLYDSL_REGISTER_TARGET_PASSES [[  mlirRegisterFlyToROCDLConversionPass();
  mlirRegisterFlyROCDLClusterAttrPass();]])

  set(FLYDSL_REGISTER_EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCPIFly
    MLIRCPIFlyROCDL
    MLIRCPILLVMOption
    MLIRCAPIArith
    MLIRCAPIGPU
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIVector
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
  )
endif()
