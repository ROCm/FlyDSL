# Bake the SHA-256 of every native library in LIBS_DIR into _native_digest.py.
#
# Usage: cmake -DLIBS_DIR=<python_packages/flydsl/_mlir/_mlir_libs> -P WriteNativeDigest.cmake
#
# The JIT disk cache keys artifacts by this table instead of hashing several
# hundred MB of libraries in every process. Symlinks are skipped (their targets
# are hashed under their own names). The file is only rewritten when a digest
# changes, so an up-to-date build leaves it untouched.

if(NOT IS_DIRECTORY "${LIBS_DIR}")
  message(FATAL_ERROR "WriteNativeDigest: LIBS_DIR '${LIBS_DIR}' is not a directory")
endif()

file(GLOB _names LIST_DIRECTORIES false RELATIVE "${LIBS_DIR}"
  "${LIBS_DIR}/*.so" "${LIBS_DIR}/*.so.*" "${LIBS_DIR}/*.a")
list(REMOVE_DUPLICATES _names)
list(SORT _names)

set(_entries "")
foreach(_name IN LISTS _names)
  if(IS_SYMLINK "${LIBS_DIR}/${_name}")
    continue()
  endif()
  file(SHA256 "${LIBS_DIR}/${_name}" _digest)
  string(APPEND _entries "    \"${_name}\": \"${_digest}\",\n")
endforeach()

set(_output "${LIBS_DIR}/_native_digest.py")
file(WRITE "${_output}.tmp"
  "\"\"\"Build-generated SHA-256 of each native library in this directory.\"\"\"\n"
  "\n"
  "NATIVE_SHA256 = {\n"
  "${_entries}"
  "}\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_output}.tmp" "${_output}")
file(REMOVE "${_output}.tmp")
