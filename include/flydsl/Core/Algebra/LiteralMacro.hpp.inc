// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// Short IntTuple literals: `_` (none), `_0` to `_32`, and the powers of two
// `_64` to `_1024`.
//
// Include this file twice: the first include defines the literals, the second
// one undefines them together with FLYDSL_CORE_LITERAL_CAT. The literal
// category defaults to `Cat`, the template parameter of the Algebra templates.
// Define FLYDSL_CORE_LITERAL_CAT before the first include to pin a concrete
// category, e.g. `#define FLYDSL_CORE_LITERAL_CAT ::mlir::fly::core::Trait`.
//
// No include guard on purpose.

#ifndef FLYDSL_CORE_LITERAL_ACTIVE
#define FLYDSL_CORE_LITERAL_ACTIVE

#ifndef FLYDSL_CORE_LITERAL_CAT
#define FLYDSL_CORE_LITERAL_CAT Cat
#endif

#define FLYDSL_CORE_LITERAL_INT(N) ::mlir::fly::core::IntTuple<FLYDSL_CORE_LITERAL_CAT>::getSInt(N)

#define _ ::mlir::fly::core::IntTuple<FLYDSL_CORE_LITERAL_CAT>::getNone()
#define _0 ::mlir::fly::core::IntTuple<FLYDSL_CORE_LITERAL_CAT>::getZero()
#define _1 ::mlir::fly::core::IntTuple<FLYDSL_CORE_LITERAL_CAT>::getOne()
#define _2 FLYDSL_CORE_LITERAL_INT(2)
#define _3 FLYDSL_CORE_LITERAL_INT(3)
#define _4 FLYDSL_CORE_LITERAL_INT(4)
#define _5 FLYDSL_CORE_LITERAL_INT(5)
#define _6 FLYDSL_CORE_LITERAL_INT(6)
#define _7 FLYDSL_CORE_LITERAL_INT(7)
#define _8 FLYDSL_CORE_LITERAL_INT(8)
#define _9 FLYDSL_CORE_LITERAL_INT(9)
#define _10 FLYDSL_CORE_LITERAL_INT(10)
#define _11 FLYDSL_CORE_LITERAL_INT(11)
#define _12 FLYDSL_CORE_LITERAL_INT(12)
#define _13 FLYDSL_CORE_LITERAL_INT(13)
#define _14 FLYDSL_CORE_LITERAL_INT(14)
#define _15 FLYDSL_CORE_LITERAL_INT(15)
#define _16 FLYDSL_CORE_LITERAL_INT(16)
#define _17 FLYDSL_CORE_LITERAL_INT(17)
#define _18 FLYDSL_CORE_LITERAL_INT(18)
#define _19 FLYDSL_CORE_LITERAL_INT(19)
#define _20 FLYDSL_CORE_LITERAL_INT(20)
#define _21 FLYDSL_CORE_LITERAL_INT(21)
#define _22 FLYDSL_CORE_LITERAL_INT(22)
#define _23 FLYDSL_CORE_LITERAL_INT(23)
#define _24 FLYDSL_CORE_LITERAL_INT(24)
#define _25 FLYDSL_CORE_LITERAL_INT(25)
#define _26 FLYDSL_CORE_LITERAL_INT(26)
#define _27 FLYDSL_CORE_LITERAL_INT(27)
#define _28 FLYDSL_CORE_LITERAL_INT(28)
#define _29 FLYDSL_CORE_LITERAL_INT(29)
#define _30 FLYDSL_CORE_LITERAL_INT(30)
#define _31 FLYDSL_CORE_LITERAL_INT(31)
#define _32 FLYDSL_CORE_LITERAL_INT(32)
#define _64 FLYDSL_CORE_LITERAL_INT(64)
#define _128 FLYDSL_CORE_LITERAL_INT(128)
#define _256 FLYDSL_CORE_LITERAL_INT(256)
#define _512 FLYDSL_CORE_LITERAL_INT(512)
#define _1024 FLYDSL_CORE_LITERAL_INT(1024)

#else

#undef _
#undef _0
#undef _1
#undef _2
#undef _3
#undef _4
#undef _5
#undef _6
#undef _7
#undef _8
#undef _9
#undef _10
#undef _11
#undef _12
#undef _13
#undef _14
#undef _15
#undef _16
#undef _17
#undef _18
#undef _19
#undef _20
#undef _21
#undef _22
#undef _23
#undef _24
#undef _25
#undef _26
#undef _27
#undef _28
#undef _29
#undef _30
#undef _31
#undef _32
#undef _64
#undef _128
#undef _256
#undef _512
#undef _1024
#undef FLYDSL_CORE_LITERAL_INT
#undef FLYDSL_CORE_LITERAL_CAT
#undef FLYDSL_CORE_LITERAL_ACTIVE

#endif
