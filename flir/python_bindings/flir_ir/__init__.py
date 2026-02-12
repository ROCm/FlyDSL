# This directory is a CMake source directory, NOT a runtime Python package.
#
# The actual _mlir_libs package initializer is _mlir_libs_init.py in this
# directory. It is copied to:
#   <build>/python_packages/flydsl/flydsl/_mlir/_mlir_libs/__init__.py
# by CMake's post-build step and by setup_python_package.sh.
#
# See _mlir_libs_init.py for the real initialization logic.
