// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// The type spellings the Fly parsers reject. A rejected type parameter also reports the
// parameter it failed to parse.

// expected-error @+2 {{incongruent layout shape and stride}}
// expected-error @+1 {{failed to parse Fly_Layout parameter}}
func.func private @f(!fly.layout<(4,8):(1)>)

// -----

// expected-error @+2 {{invalid swizzle parameters}}
// expected-error @+1 {{failed to parse Fly_Swizzle parameter}}
func.func private @f(!fly.swizzle<S<-1,3,3>>)

// -----

// expected-error @+1 {{failed to parse Fly_CoordSwizzle parameter}}
func.func private @f(!fly.coord_swizzle<CS<-1,0,[],0,[]>>)

// -----

// expected-error @+2 {{ratio numerator exceeds the representable range}}
// expected-error @+1 {{failed to parse Fly_IntTuple parameter}}
func.func private @f(!fly.int_tuple<3000000000/7>)

// -----

// expected-error @+2 {{invalid or unrepresentable ratio}}
// expected-error @+1 {{failed to parse Fly_IntTuple parameter}}
func.func private @f(!fly.int_tuple<1/0>)

// -----

// expected-error @+2 {{divisibility must be positive}}
// expected-error @+1 {{failed to parse Fly_IntTuple parameter}}
func.func private @f(!fly.int_tuple<?{div=0}>)

// -----

// expected-error @+2 {{basis mode exceeds the representable range}}
// expected-error @+1 {{failed to parse Fly_IntTuple parameter}}
func.func private @f(!fly.int_tuple<1E256>)

// -----

// expected-error @+2 {{invalid basis coefficient or modes}}
// expected-error @+1 {{failed to parse Fly_IntTuple parameter}}
func.func private @f(!fly.int_tuple<*E0>)

// -----

// A basis type holds a basis leaf.
// expected-error @+1 {{expected a basis leaf}}
func.func private @f(!fly.basis<4>)

// -----

// A tile mode is an extent; a ratio has no tile meaning.
// expected-error @+1 {{failed to parse Fly_Tile parameter}}
func.func private @f(!fly.tile<[1/2|*]>)

// -----

// expected-error @+1 {{alignment must be a positive multiple of element byte size (4), got 2}}
func.func private @f(!fly.ptr<f32, global, align<2>>)
