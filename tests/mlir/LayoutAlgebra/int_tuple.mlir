// RUN: %fly-opt %s | FileCheck %s

// Tests for IntTuple arithmetic operations:
//   fly.int_tuple_add, fly.int_tuple_sub, fly.int_tuple_mul,
//   fly.int_tuple_div, fly.int_tuple_mod,
//   fly.int_tuple_product, fly.int_tuple_product_each,
//   fly.shape_div, fly.ceil_div, fly.elem_less, fly.equal

// -----

// CHECK-LABEL: @test_int_tuple_add
func.func @test_int_tuple_add() -> !fly.int_tuple<(6, 10)> {
  // (4,8) + (2,2) = (6,10)
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %b = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  // CHECK: fly.int_tuple_add(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_add(%a, %b) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(2, 2)>) -> !fly.int_tuple<(6, 10)>
  return %result : !fly.int_tuple<(6, 10)>
}

// CHECK-LABEL: @test_int_tuple_sub
func.func @test_int_tuple_sub() -> !fly.int_tuple<(2, 6)> {
  // (4,8) - (2,2) = (2,6)
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %b = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  // CHECK: fly.int_tuple_sub(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_sub(%a, %b) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(2, 2)>) -> !fly.int_tuple<(2, 6)>
  return %result : !fly.int_tuple<(2, 6)>
}

// CHECK-LABEL: @test_int_tuple_mul
func.func @test_int_tuple_mul() -> !fly.int_tuple<(8, 16)> {
  // (4,8) * (2,2) = (8,16)
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %b = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  // CHECK: fly.int_tuple_mul(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_mul(%a, %b) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(2, 2)>) -> !fly.int_tuple<(8, 16)>
  return %result : !fly.int_tuple<(8, 16)>
}

// CHECK-LABEL: @test_int_tuple_div
func.func @test_int_tuple_div() -> !fly.int_tuple<(2, 4)> {
  // (4,8) / (2,2) = (2,4)
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %b = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  // CHECK: fly.int_tuple_div(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_div(%a, %b) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(2, 2)>) -> !fly.int_tuple<(2, 4)>
  return %result : !fly.int_tuple<(2, 4)>
}

// CHECK-LABEL: @test_int_tuple_mod
func.func @test_int_tuple_mod() -> !fly.int_tuple<(?, ?)> {
  // mod result type is dynamic (inferred as (?,?))
  %a = fly.static {elems = [5 : i32, 8 : i32]} : () -> !fly.int_tuple<(5, 8)>
  %b = fly.static {elems = [2 : i32, 4 : i32]} : () -> !fly.int_tuple<(2, 4)>
  // CHECK: fly.int_tuple_mod(%{{.*}}, %{{.*}})
  %result = fly.int_tuple_mod(%a, %b) : (!fly.int_tuple<(5, 8)>, !fly.int_tuple<(2, 4)>) -> !fly.int_tuple<(?, ?)>
  return %result : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_int_tuple_product
func.func @test_int_tuple_product() -> !fly.int_tuple<32> {
  // product((4,8)) = 4 * 8 = 32
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  // CHECK: fly.int_tuple_product(%{{.*}})
  %result = fly.int_tuple_product(%a) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<32>
  return %result : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_int_tuple_product_each
func.func @test_int_tuple_product_each(%input: !fly.int_tuple<((2, 3), (4, 5))>) -> !fly.int_tuple<(6, 20)> {
  // product_each(((2,3),(4,5))) = (6, 20)
  // CHECK: fly.int_tuple_product_each(%{{.*}})
  %result = fly.int_tuple_product_each(%input) : (!fly.int_tuple<((2, 3), (4, 5))>) -> !fly.int_tuple<(6, 20)>
  return %result : !fly.int_tuple<(6, 20)>
}

// CHECK-LABEL: @test_shape_div
func.func @test_shape_div() -> !fly.int_tuple<(2, 4)> {
  // shape_div: (4,8) / (2,2) = (2,4)
  %a = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %b = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  // CHECK: fly.shape_div(%{{.*}}, %{{.*}})
  %result = fly.shape_div(%a, %b) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(2, 2)>) -> !fly.int_tuple<(2, 4)>
  return %result : !fly.int_tuple<(2, 4)>
}

// CHECK-LABEL: @test_ceil_div
func.func @test_ceil_div() -> !fly.int_tuple<(3, 3)> {
  // ceil_div: (5,7) / (2,3) = (3,3)
  %a = fly.static {elems = [5 : i32, 7 : i32]} : () -> !fly.int_tuple<(5, 7)>
  %b = fly.static {elems = [2 : i32, 3 : i32]} : () -> !fly.int_tuple<(2, 3)>
  // CHECK: fly.ceil_div(%{{.*}}, %{{.*}})
  %result = fly.ceil_div(%a, %b) : (!fly.int_tuple<(5, 7)>, !fly.int_tuple<(2, 3)>) -> !fly.int_tuple<(3, 3)>
  return %result : !fly.int_tuple<(3, 3)>
}

// CHECK-LABEL: @test_dynamic_int_tuple_add
func.func @test_dynamic_int_tuple_add(%a: i32, %b: i32, %c: i32, %d: i32) -> !fly.int_tuple<(?, ?)> {
  %lhs = fly.make_int_tuple(%a, %b) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  %rhs = fly.make_int_tuple(%c, %d) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  // CHECK: fly.int_tuple_add
  %result = fly.int_tuple_add(%lhs, %rhs) : (!fly.int_tuple<(?, ?)>, !fly.int_tuple<(?, ?)>) -> !fly.int_tuple<(?, ?)>
  return %result : !fly.int_tuple<(?, ?)>
}
