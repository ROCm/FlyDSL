// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Preprocessor FOR_EACH utility for iterating over FLYDSL_BACKENDS_TUPLE.
// Adapted from Triton (python/src/main.cc).
//
// Usage:
//   // FLYDSL_BACKENDS_TUPLE is defined by CMake as e.g. (rocdl)
//   #define MY_ACTION(name) void do_##name();
//   FOR_EACH_BACKEND(MY_ACTION, FLYDSL_BACKENDS_TUPLE)
//   // expands to: void do_rocdl();

#ifndef FLYDSL_BACKEND_FOREACHBACKEND_H
#define FLYDSL_BACKEND_FOREACHBACKEND_H

#define _FLYDSL_FOR_EACH_1(MACRO, X) MACRO(X)
#define _FLYDSL_FOR_EACH_2(MACRO, X, ...) MACRO(X) _FLYDSL_FOR_EACH_1(MACRO, __VA_ARGS__)
#define _FLYDSL_FOR_EACH_3(MACRO, X, ...) MACRO(X) _FLYDSL_FOR_EACH_2(MACRO, __VA_ARGS__)
#define _FLYDSL_FOR_EACH_4(MACRO, X, ...) MACRO(X) _FLYDSL_FOR_EACH_3(MACRO, __VA_ARGS__)
#define _FLYDSL_FOR_EACH_5(MACRO, X, ...) MACRO(X) _FLYDSL_FOR_EACH_4(MACRO, __VA_ARGS__)

#define _FLYDSL_FOR_EACH_NARG(...) _FLYDSL_FOR_EACH_NARG_(__VA_ARGS__, _FLYDSL_FOR_EACH_RSEQ_N())
#define _FLYDSL_FOR_EACH_NARG_(...) _FLYDSL_FOR_EACH_ARG_N(__VA_ARGS__)
#define _FLYDSL_FOR_EACH_ARG_N(_1, _2, _3, _4, _5, N, ...) N
#define _FLYDSL_FOR_EACH_RSEQ_N() 5, 4, 3, 2, 1, 0

#define _FLYDSL_CONCAT(x, y) _FLYDSL_CONCAT1(x, y)
#define _FLYDSL_CONCAT1(x, y) x##y

#define _FLYDSL_FOR_EACH(MACRO, ...)                                                               \
  _FLYDSL_CONCAT(_FLYDSL_FOR_EACH_, _FLYDSL_FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define _FLYDSL_FOR_EACH_NARG_HELPER(...) _FLYDSL_FOR_EACH_NARG(__VA_ARGS__)

#define _FLYDSL_REMOVE_PARENS(...) __VA_ARGS__

#define _FLYDSL_FOR_EACH_INTERMEDIATE(MACRO, ...) _FLYDSL_FOR_EACH(MACRO, __VA_ARGS__)

#define FOR_EACH_BACKEND(MACRO, ARGS_WITH_PARENS)                                                  \
  _FLYDSL_FOR_EACH_INTERMEDIATE(MACRO, _FLYDSL_REMOVE_PARENS ARGS_WITH_PARENS)

#endif // FLYDSL_BACKEND_FOREACHBACKEND_H
