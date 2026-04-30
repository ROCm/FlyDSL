# Adapted from https://github.com/deepseek-ai/FlashMLA/tree/main/tests/test_flash_mla_sparse_decoding.py
import dataclasses
from typing import Tuple, List

import rich.console
import rich.table

import torch

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    correctness_cases = []
    corner_cases = []
    for d_qk in [576, 512]:
        for have_extra_k in ([False, True] if d_qk == 512 else [False]):
            for have_extra_topk_len in ([False, True] if have_extra_k else [False]):
                for have_topk_len in ([False, True] if d_qk == 512 else [False]):
                    for h_q in [64, 128]:
                        cur_correctness_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=True,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0)
                            for (s_k, topk, block_size) in [
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [1, 3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                        ]
                        correctness_cases.extend(cur_correctness_cases)

                        cur_corner_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        is_all_indices_invalid=is_all_indices_invalid,
                                        have_zero_seqlen_k=have_zero_seqlen_k,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=enable_attn_sink,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0,
                            )
                            for (s_k, topk, block_size) in [
                                (512, 64, 61),
                                (650, 576, 53),
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 61),
                                (650, 576, 53),
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                            for is_all_indices_invalid in [True, False]
                            for have_zero_seqlen_k in [True, False]
                            for enable_attn_sink in [True, False]
                            if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                        ]
                        corner_cases.extend(cur_corner_cases)

    base_and_bszs = [
        # V3.2
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128]),
        # MODEL1 CONFIG1
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG2
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG3
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG4
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
    ]
    performance_cases = [
        # Production cases
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ] + [
        # Peak perf cases
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=d_qk)
        for h_q in [64, 128]
        for d_qk in [512, 576]
    ]

    return correctness_cases + corner_cases + performance_cases


@dataclasses.dataclass
class Result:
    is_correct: bool

_counter = 0


def _check_is_allclose(name: str, actual: torch.Tensor, expected: torch.Tensor, abs_tol: float, rel_tol: float) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    if finite.any():
        diff = (actual_f[finite] - expected_f[finite]).abs()
        tol = abs_tol + rel_tol * expected_f[finite].abs()
        is_correct = bool((diff <= tol).all().item())
        max_diff = diff.max().item()
    else:
        is_correct = True
        max_diff = 0.0
    is_correct &= bool(torch.equal(torch.isposinf(actual_f), torch.isposinf(expected_f)))
    is_correct &= bool(torch.equal(torch.isneginf(actual_f), torch.isneginf(expected_f)))
    if not is_correct:
        print(f"{name} mismatch: max_diff={max_diff}")
    return is_correct


@torch.inference_mode()
def test_flash_mla(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter
        _counter += 1
    assert p.decode

    print("================")
    print(f"Running reference-only on {p}")

    t = lib.generate_testcase_for_decode(p)

    out_ans, lse_ans = ref.ref_sparse_attn_decode(p, t)
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    is_out_correct = _check_is_allclose("out", out_ans, out_ref, abs_tol=1e-3, rel_tol=2.01/128)
    is_lse_correct = _check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
    is_correct = is_out_correct and is_lse_correct
    return Result(is_correct)


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    print(f"{len(testcases)} reference-only correctness cases to run")

    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        result = test_flash_mla(testcase)
        results.append((testcase, result))
        if not result.is_correct:
            failed_cases.append(testcase)
            import sys
            sys.exit(1)

    console = rich.console.Console(width=120)
    table = rich.table.Table(show_header=True, header_style="bold cyan")
    table.add_column("topk")
    table.add_column("Bsz")
    table.add_column("h_q&k")
    table.add_column("sq")
    table.add_column("sk")
    table.add_column("d_qk")
    table.add_column("Feats")
    table.add_column(" ")

    for testcase, result in results:
        assert testcase.decode
        topk_str = f"{testcase.topk}" if testcase.decode.extra_topk is None else f"{testcase.topk}+{testcase.decode.extra_topk}"
        table.add_row(
            topk_str,
            str(testcase.decode.b),
            f"{testcase.h_q:3d} {testcase.h_kv}",
            str(testcase.s_q),
            str(testcase.s_kv),
            str(testcase.d_qk),
            " V"[testcase.decode.is_varlen] + " L"[testcase.have_topk_length] + " E"[testcase.decode.have_extra_topk_length],
            "" if result.is_correct else "X"
        )
    console.print(table)

    num_correct_testcases = [result.is_correct for t, result in results if t.check_correctness].count(True)
    num_correctness_cases = sum([1 for t in testcases if t.check_correctness])
    if num_correct_testcases == num_correctness_cases:
        print(f"{num_correct_testcases}/{num_correctness_cases} correctness cases passed")
    else:
        print(f"{num_correct_testcases}/{num_correctness_cases} correctness cases passed")
        for t in failed_cases:
            print(f"\t{t},")
    

if __name__ == "__main__":
    main()
