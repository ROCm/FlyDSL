# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .kernel_pipeline_shared import KVReg, KInstr, KImm


class _FloorExpressionOps:
    def _emit_floor(self, expr) -> KVReg:
        """
        Emit a floor expression.

        Handles:
        - floor(x / n) where n is power-of-2 -> shift right
        - floor(Add(...)) with fractions -> find common denominator and shift
        - floor(floor(...)) -> emit inner floor (nested floors are no-op)
        - floor(integer) -> no-op
        - floor(symbol) -> no-op (symbols are integers in our context)
        """
        import sympy
        from sympy import floor, Mul, Add, Integer

        inner = expr.args[0]

        # floor(Mul(...)) - floor division pattern
        if isinstance(inner, Mul):
            return self._emit_floor_div(inner)

        # floor(Add(...)) - need to handle terms with fractions
        if isinstance(inner, Add):
            return self._emit_floor_add(inner)

        # floor(floor(...)) - just emit the inner floor
        if getattr(inner, "func", None) == floor:
            return self._emit_floor(inner)

        # floor(Integer) - just return the integer
        if isinstance(inner, Integer):
            return self.get_or_emit(int(inner))

        # floor(Symbol) - symbols are integers, no-op
        if isinstance(inner, sympy.Symbol):
            return self.get_or_emit(inner)

        # floor(Mod) - emit the mod (Mod produces integers)
        if isinstance(inner, sympy.Mod):
            return self.get_or_emit(inner)

        raise NotImplementedError(f"floor expression not supported: {expr}")

    def _emit_floor_div(self, mul_expr) -> KVReg:
        """
        Emit floor(x * 1/n) = floor(x/n).

        Parses a Mul expression to find the numerator and divisor,
        then emits a right shift if divisor is power-of-2.
        """
        from sympy import Integer, Rational, Pow

        x_part = None
        divisor = None
        const_coeff = 1

        for factor in mul_expr.args:
            if isinstance(factor, Pow) and factor.args[1] == -1:
                # x^(-1) = 1/x
                divisor = factor.args[0]
            elif isinstance(factor, Rational) and not isinstance(factor, Integer):
                # p/q rational
                if factor.p != 1:
                    const_coeff *= int(factor.p)
                divisor = Integer(factor.q)
            elif isinstance(factor, Integer):
                const_coeff *= int(factor)
            else:
                if x_part is None:
                    x_part = factor
                else:
                    x_part = x_part * factor

        if x_part is None:
            # Pure constant division
            if divisor is not None and isinstance(divisor, Integer):
                return self.get_or_emit(const_coeff // int(divisor))
            return self.get_or_emit(const_coeff)

        if divisor is None:
            # No division, just emit the expression
            if const_coeff != 1:
                return self.get_or_emit(x_part * Integer(const_coeff))
            return self.get_or_emit(x_part)

        if not isinstance(divisor, Integer):
            raise NotImplementedError(f"Non-integer divisor not supported: {divisor}")

        div_val = int(divisor)
        if div_val <= 0 or (div_val & (div_val - 1)) != 0:
            raise NotImplementedError(f"Floor divisor must be power-of-2: {div_val}")

        # Emit x * const_coeff first
        if const_coeff == 1:
            num_reg = self.get_or_emit(x_part)
        else:
            num_reg = self.get_or_emit(x_part * Integer(const_coeff))

        # Then shift right
        shift = div_val.bit_length() - 1
        if shift > 0:
            result = self.ctx.vreg()
            self.ctx.program.emit(
                KInstr(
                    "v_lshrrev_b32",
                    (result,),
                    (KImm(shift), num_reg),
                    comment=f"floor div by {div_val} (shift)",
                )
            )
            return result

        return num_reg

    def _emit_floor_add(self, add_expr) -> KVReg:
        """
        Emit floor(Add(...)) with fractional terms.

        Algorithm:
        1. Parse each term to find numerator and denominator
        2. Find LCM of all denominators
        3. Scale each numerator by LCM/denominator
        4. Sum all scaled numerators
        5. Shift right by log2(LCM)

        Examples:
        - floor(a/16 + 1/2) = floor((a + 8)/16) = (a + 8) >> 4
        - floor(tid_y/2 + floor(tid_x/8)/32) = (16*tid_y + floor(tid_x/8)) >> 5
        """
        import sympy
        from sympy import Integer, Rational, Mul, floor
        from math import gcd

        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        def get_num_denom(term):
            """Extract (numerator_expr, denominator) from a term."""
            # Pure integer
            if isinstance(term, Integer):
                return (Integer(int(term)), 1)

            # Pure rational like 1/2
            if isinstance(term, Rational) and not isinstance(term, Integer):
                return (Integer(int(term.p)), int(term.q))

            # Symbol (integer)
            if isinstance(term, sympy.Symbol):
                return (term, 1)

            # floor(...) - produces integer
            if getattr(term, "func", None) == floor:
                return (term, 1)

            # Mod - produces integer
            if isinstance(term, sympy.Mod):
                return (term, 1)

            # Mul - look for rational factor
            if isinstance(term, Mul):
                divisor = 1
                numerator_parts = []

                for factor in term.args:
                    if isinstance(factor, Rational) and not isinstance(factor, Integer):
                        # p/q rational
                        if factor.p != 1:
                            numerator_parts.append(Integer(int(factor.p)))
                        divisor = int(factor.q)
                    elif isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                        # x^(-1) = 1/x
                        if isinstance(factor.args[0], Integer):
                            divisor = int(factor.args[0])
                        else:
                            raise NotImplementedError(f"Non-integer inverse: {factor}")
                    else:
                        numerator_parts.append(factor)

                if numerator_parts:
                    if len(numerator_parts) == 1:
                        return (numerator_parts[0], divisor)
                    else:
                        return (Mul(*numerator_parts), divisor)
                else:
                    return (Integer(1), divisor)

            # Default: treat as integer
            return (term, 1)

        # Parse all terms
        terms = []
        for term in add_expr.args:
            num, denom = get_num_denom(term)
            terms.append((num, denom))

        # Find LCM of all denominators
        common_denom = 1
        for _, denom in terms:
            common_denom = lcm(common_denom, denom)

        # If all denominators are 1, just emit the add (no floor needed)
        if common_denom == 1:
            return self.get_or_emit(add_expr)

        # Check LCM is power of 2
        if common_denom <= 0 or (common_denom & (common_denom - 1)) != 0:
            raise NotImplementedError(
                f"Floor add requires power-of-2 LCM: {common_denom}"
            )

        # Build scaled sum: sum of (num * (common_denom / denom))
        scaled_sum = None
        for num, denom in terms:
            scale = common_denom // denom
            if scale == 1:
                scaled_term = num
            elif isinstance(num, Integer):
                scaled_term = Integer(int(num) * scale)
            else:
                scaled_term = num * Integer(scale)

            if scaled_sum is None:
                scaled_sum = scaled_term
            else:
                scaled_sum = scaled_sum + scaled_term

        # Emit the scaled sum
        sum_reg = self.get_or_emit(scaled_sum)

        # Shift right by log2(common_denom)
        shift = common_denom.bit_length() - 1
        result = self.ctx.vreg()
        self.ctx.program.emit(
            KInstr(
                "v_lshrrev_b32",
                (result,),
                (KImm(shift), sum_reg),
                comment=f"floor add by {common_denom} (shift)",
            )
        )

        return result
