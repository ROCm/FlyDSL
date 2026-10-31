#!/usr/bin/env python3

"""
Pure Python repro for `for` loop variable behavior.
"""


def case_1_plain_for():
    i = 999
    for i in range(5):
        pass
    print("[case1] plain for -> i =", i, "(expect 4)")


def case_2_for_with_i_plus_1():
    i = 999
    for i in range(5):
        i = i + 1
    print("[case2] for + i=i+1 -> i =", i, "(expect 5)")


def case_3_empty_loop_keeps_old_value():
    i = 999
    for i in range(0):
        i = i + 1
    print("[case3] empty loop -> i =", i, "(expect 999)")


if __name__ == "__main__":
    case_1_plain_for()
    case_2_for_with_i_plus_1()
    case_3_empty_loop_keeps_old_value()
