#!/usr/bin/env python3

"""FlyDSL version of lane-capture nested-function demo."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def lane_capture_kernel():
    tidx = fx.thread_idx.x
    lane = tidx & fx.Int32(31)

    def choose_value(x):
        out = x
        # Directly capture/use outer variable `lane`.
        if lane == fx.Int32(0):
            out = fx.Int32(123)
        else:
            out = fx.Int32(456)
        return out

    v = choose_value(fx.Int32(1))
    if tidx <= fx.Int32(1):
        fx.printf("capture: tidx={} lane={} v={}", tidx, lane, v)


@flyc.kernel
def lane_capture_in_if_body_kernel():
    tidx = fx.thread_idx.x
    lane = tidx & fx.Int32(31)

    def compute_with_lane(x):
        out = x
        # lane is used in branch body as a value, not only in predicate.
        if tidx == fx.Int32(0):
            out = lane + fx.Int32(1000)
        else:
            out = lane + fx.Int32(2000)
        return out

    v = compute_with_lane(fx.Int32(1))
    if tidx <= fx.Int32(1):
        fx.printf("body-use: tidx={} lane={} v={}", tidx, lane, v)


@flyc.kernel
def lane_modify_in_if_kernel():
    tidx = fx.thread_idx.x
    lane = tidx & fx.Int32(31)

    def modify_lane():
        lane = fx.Int32(1)
        fx.printf("modify: tidx={} lane={}", tidx, lane)
        return

    modify_lane()
    if tidx <= fx.Int32(3):
        fx.printf("modify: tidx={} lane={}", tidx, lane)


@flyc.jit
def run_case(stream: fx.Stream = fx.Stream(None)):
    lane_capture_kernel().launch(grid=(1, 1, 1), block=(2, 1, 1), stream=stream)
    lane_capture_in_if_body_kernel().launch(grid=(1, 1, 1), block=(2, 1, 1), stream=stream)


@flyc.jit
def run_modify_case(stream: fx.Stream = fx.Stream(None)):
    lane_modify_in_if_kernel().launch(grid=(1, 1, 1), block=(10, 1, 1), stream=stream)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required.")

    stream = torch.cuda.current_stream()
    # print("=== run_case ===")
    # run_case(stream)
    # torch.cuda.synchronize()

    print("=== run_modify_case ===")
    run_modify_case(stream)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
