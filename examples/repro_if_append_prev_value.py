#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
最小复现：for + if 中变量在未进入 if 时的取值行为。

包含三种场景：
1) 首轮不进 if -> UnboundLocalError
2) 首轮进 if，后续不进 if -> 复用上一轮值
3) 每轮显式初始化默认值 -> 不复用上一轮
"""


def case_first_iter_not_enter_if():
    vals = []
    for i in range(3):
        if i > 0:  # i=0 不进
            x = f"new@{i}"
        vals.append(x)
    return vals


def case_reuse_prev_value():
    vals = []
    for i in range(4):
        if i == 0 or i == 3:  # i=1,2 不进 if
            x = f"new@{i}"
        vals.append(x)
    return vals


def case_explicit_default():
    vals = []
    for i in range(4):
        x = "default"
        if i == 0 or i == 3:
            x = f"new@{i}"
        vals.append(x)
    return vals


def main():
    print("case1: 首次不进 if（预期报 UnboundLocalError）")
    try:
        print(case_first_iter_not_enter_if())
    except Exception as e:
        print(type(e).__name__, e)

    print("\ncase2: 首次进 if，后续不进 if（复用上一轮值）")
    print(case_reuse_prev_value())

    print("\ncase3: 每轮先给默认值（不复用上一轮值）")
    print(case_explicit_default())


if __name__ == "__main__":
    main()

