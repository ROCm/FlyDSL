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
