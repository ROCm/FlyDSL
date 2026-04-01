"""Tests for known_block_size attribute on @flyc.kernel."""

import re

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx


# ---------------------------------------------------------------------------
# Kernels with various known_block_size values
# ---------------------------------------------------------------------------

@flyc.kernel(known_block_size=[512, 1, 1])
def _kn_bs512(x: fx.Tensor):
    pass


@flyc.kernel(known_block_size=[64, 1, 1])
def _kn_bs64(x: fx.Tensor):
    pass


@flyc.kernel(known_block_size=[32, 1, 1])
def _kn_bs32(x: fx.Tensor):
    pass


@flyc.kernel(known_block_size=[128, 1, 2])
def _kn_bs128_1_2(x: fx.Tensor):
    pass


@flyc.kernel(known_block_size=[128, 4, 2])
def _kn_bs128_4_2(x: fx.Tensor):
    pass


@flyc.kernel
def _kn_no_block_size(x: fx.Tensor):
    pass


# ---------------------------------------------------------------------------
# JIT launchers
# ---------------------------------------------------------------------------

@flyc.jit
def _launch_bs512(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs512(x).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream)


@flyc.jit
def _launch_bs64(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs64(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _launch_bs32(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs32(x).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@flyc.jit
def _launch_bs128_1_2(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs128_1_2(x).launch(grid=(1, 1, 1), block=(128, 1, 2), stream=stream)


@flyc.jit
def _launch_bs128_4_2(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs128_4_2(x).launch(grid=(1, 1, 1), block=(128, 4, 2), stream=stream)


@flyc.jit
def _launch_no_block_size(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_source_ir(launch_fn, x):
    """Call the JIT function once, then return the source IR string."""
    launch_fn(x, stream=torch.cuda.current_stream())
    # Retrieve the most recently cached CompiledArtifact.
    assert launch_fn._mem_cache, "expected at least one cached compilation"
    artifact = next(iter(launch_fn._mem_cache.values()))
    return artifact.source_ir


def _get_compiled_ir(launch_fn, x):
    """Call the JIT function once, then return the compiled IR string."""
    launch_fn(x, stream=torch.cuda.current_stream())
    assert launch_fn._mem_cache, "expected at least one cached compilation"
    artifact = next(iter(launch_fn._mem_cache.values()))
    return artifact.ir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKnownBlockSize:
    """Verify that known_block_size is emitted in IR and affects metadata."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = torch.zeros(64, device="cuda", dtype=torch.float32)

    @pytest.mark.parametrize(
        "launch_fn, expected",
        [
            (_launch_bs512, [512, 1, 1]),
            (_launch_bs64, [64, 1, 1]),
            (_launch_bs32, [32, 1, 1]),
            (_launch_bs128_1_2, [128, 1, 2]),
            (_launch_bs128_4_2, [128, 4, 2]),
        ],
        ids=["512x1x1", "64x1x1", "32x1x1", "128x1x2", "128x4x2"],
    )
    def test_source_ir_contains_known_block_size(self, launch_fn, expected):
        source_ir = _get_source_ir(launch_fn, self.x)
        attr_str = f"known_block_size = array<i32: {expected[0]}, {expected[1]}, {expected[2]}>"
        assert attr_str in source_ir, (
            f"expected '{attr_str}' in source IR, got:\n{source_ir}"
        )

    @pytest.mark.parametrize(
        "launch_fn, expected",
        [
            (_launch_bs512, [512, 1, 1]),
            (_launch_bs64, [64, 1, 1]),
            (_launch_bs32, [32, 1, 1]),
            (_launch_bs128_1_2, [128, 1, 2]),
            (_launch_bs128_4_2, [128, 4, 2]),
        ],
        ids=["512x1x1", "64x1x1", "32x1x1", "128x1x2", "128x4x2"],
    )
    def test_compiled_ir_has_max_flat_workgroup_size(self, launch_fn, expected):
        compiled_ir = _get_compiled_ir(launch_fn, self.x)
        total_threads = expected[0] * expected[1] * expected[2]
        # The compiled IR should report max_flat_workgroup_size >= total_threads
        match = re.search(r"max_flat_workgroup_size\s*=\s*(\d+)", compiled_ir)
        assert match is not None, (
            f"max_flat_workgroup_size not found in compiled IR:\n{compiled_ir}"
        )
        max_wg = int(match.group(1))
        assert max_wg >= total_threads, (
            f"max_flat_workgroup_size={max_wg} < total_threads={total_threads}"
        )

    def test_no_known_block_size_omitted_from_ir(self):
        source_ir = _get_source_ir(_launch_no_block_size, self.x)
        # Check for the attribute syntax, not just the substring (which may
        # appear in kernel names like "_kn_no_block_size_0").
        assert "known_block_size = array<i32:" not in source_ir, (
            f"known_block_size attribute should not appear when not specified:\n{source_ir}"
        )

    @pytest.mark.parametrize(
        "launch_fn, block_size",
        [
            (_launch_bs512, 512),
            (_launch_bs64, 64),
            (_launch_bs32, 32),
            (_launch_bs128_4_2, 1024),
        ],
        ids=["512", "64", "32", "1024"],
    )
    def test_kernel_launches_successfully(self, launch_fn, block_size):
        """Ensure the kernel actually launches without hipErrorLaunchFailure."""
        launch_fn(self.x, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # would raise if launch failed


class TestKnownBlockSizeValidation:
    """Verify that invalid known_block_size values are rejected early."""

    def test_wrong_length_2(self):
        with pytest.raises(ValueError, match="exactly 3 elements"):
            @flyc.kernel(known_block_size=[256, 1])
            def _bad(x: fx.Tensor):
                pass

    def test_wrong_length_4(self):
        with pytest.raises(ValueError, match="exactly 3 elements"):
            @flyc.kernel(known_block_size=[64, 1, 1, 1])
            def _bad(x: fx.Tensor):
                pass

    def test_not_a_sequence(self):
        with pytest.raises(TypeError, match="sequence of 3 positive integers"):
            @flyc.kernel(known_block_size=512)
            def _bad(x: fx.Tensor):
                pass

    def test_non_int_element(self):
        with pytest.raises(TypeError, match="must be an int"):
            @flyc.kernel(known_block_size=[64.0, 1, 1])
            def _bad(x: fx.Tensor):
                pass

    def test_zero_element(self):
        with pytest.raises(ValueError, match="must be positive"):
            @flyc.kernel(known_block_size=[0, 1, 1])
            def _bad(x: fx.Tensor):
                pass

    def test_negative_element(self):
        with pytest.raises(ValueError, match="must be positive"):
            @flyc.kernel(known_block_size=[64, -1, 1])
            def _bad(x: fx.Tensor):
                pass

    def test_none_is_accepted(self):
        """None means 'omit attribute' — should not raise."""
        @flyc.kernel(known_block_size=None)
        def _ok(x: fx.Tensor):
            pass


class TestKnownBlockSizeLaunchMismatch:
    """Verify that a warning is emitted when launch block != known_block_size."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = torch.zeros(64, device="cuda", dtype=torch.float32)

    def test_matching_block_no_warning(self):
        import warnings

        @flyc.kernel(known_block_size=[256, 1, 1])
        def _kn_match(x: fx.Tensor):
            pass

        @flyc.jit
        def _launch_match(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            _kn_match(x).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _launch_match(self.x, stream=torch.cuda.current_stream())

    def test_no_known_block_size_no_warning(self):
        import warnings

        @flyc.kernel
        def _kn_none(x: fx.Tensor):
            pass

        @flyc.jit
        def _launch_any(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            _kn_none(x).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _launch_any(self.x, stream=torch.cuda.current_stream())
