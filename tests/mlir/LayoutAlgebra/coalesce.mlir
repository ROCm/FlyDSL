// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// RUN: %fly-opt %s | FileCheck %s

// PyIR-aligned coalesce tests from tests/pyir/test_layout_algebra.py

// CHECK-LABEL: @pyir_coalesce_basic
func.func @pyir_coalesce_basic() -> !fly.layout<27 : 1> {
  %s = fly.static : !fly.int_tuple<(3, 1, 9)>
  %d = fly.static : !fly.int_tuple<(1, 9, 3)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(3, 1, 9)>, !fly.int_tuple<(1, 9, 3)>) -> !fly.layout<(3, 1, 9) : (1, 9, 3)>
  // CHECK: fly.coalesce
  %result = fly.coalesce(%layout) : (!fly.layout<(3, 1, 9) : (1, 9, 3)>) -> !fly.layout<27 : 1>
  return %result : !fly.layout<27 : 1>
}
