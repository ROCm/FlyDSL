import flydsl.compiler as flyc
import flydsl.expr as fx


# def helper_with_range(loop_count: fx.Int32):
#     for i in range(loop_count):
#         fx.printf("helper i={}", i)


@flyc.kernel
def call_helper_kernel(loop_count: fx.Int32):
    fx.printf("kernel loop_count={}", loop_count)
    # helper_with_range(loop_count)
    for i in range(loop_count):
        fx.printf("helper i={}", i)


@flyc.jit
def run_case(loop_count: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    call_helper_kernel(loop_count).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


if __name__ == "__main__":
    run_case(4)
