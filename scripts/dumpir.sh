#!/bin/bash

# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# dumpir.sh <script> <arguments>
# Example: bash scripts/dumpir.sh python tests/kernels/benchmark/matrixTranspose.py
export FLIR_DUMP_IR=1
export FLIR_REBUILD=1
exec "$@"
