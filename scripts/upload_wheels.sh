#!/usr/bin/env bash

# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -e

python -m twine upload $@

