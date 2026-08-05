"""Tests for known_block_size attribute on @flyc.kernel."""

import re

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

# ---------------------------------------------------------------------------
# Kernels with various known_block_size values
# ---------------------------------------------------------------------------


@flyc.kernel(known_block_size=[64, 1, 1])
def _kn_bs64(x: fx.Tensor):
    pass


@flyc.kernel(known_block_size=[128, 4, 2])
def _kn_bs128_4_2(x: fx.Tensor):
    pass


@flyc.kernel
def _kn_no_block_size(x: fx.Tensor):
    pass


@flyc.kernel(name="_kn_named_block_specialization")
def _kn_named_block_specialization(x: fx.Tensor):
    pass


# ---------------------------------------------------------------------------
# JIT launchers
# ---------------------------------------------------------------------------


@flyc.jit
def _launch_bs64(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs64(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _launch_bs128_4_2(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_bs128_4_2(x).launch(grid=(1, 1, 1), block=(128, 4, 2), stream=stream)


@flyc.jit
def _launch_no_block_size(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _launch_static_numeric_dims(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _kn_no_block_size(x).launch(grid=fx.Int64(1), block=fx.Int32(64), stream=stream)
    _kn_no_block_size(x).launch(
        grid=(fx.Int32(1), fx.Int64(1), fx.Int32(1)),
        block=(fx.Int64(32), fx.Int32(2), fx.Int64(1)),
        stream=stream,
    )


@flyc.jit
def _launch_dynamic_blocks(
    x: fx.Tensor,
    block_i32: fx.Int32,
    block_i64: fx.Int64,
    stream: fx.Stream = fx.Stream(None),
):
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=block_i32, stream=stream)
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=(block_i64, 1, 1), stream=stream)
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=block_i32.ir_value(), stream=stream)
    _kn_no_block_size(x).launch(grid=(1, 1, 1), block=(fx.Index(block_i32).ir_value(), 1, 1), stream=stream)


@flyc.jit
def _launch_explicit_size_with_dynamic_block(
    x: fx.Tensor,
    block_size: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    _kn_bs64(x).launch(grid=(1, 1, 1), block=(block_size, 1, 1), stream=stream)


@flyc.jit
def _launch_pending_twice(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    pending = _kn_no_block_size(x)
    pending.launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    pending.launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


@flyc.jit
def _launch_pending_same_size_twice(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    pending = _kn_no_block_size(x)
    pending.launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    pending.launch(grid=(2, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _launch_named_pending_twice(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    pending = _kn_named_block_specialization(x)
    pending.launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    pending.launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_source_ir(launch_fn, *args):
    """Call the JIT function once, then return the source IR string."""
    launch_fn(*args, stream=torch.cuda.current_stream())
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
            (_launch_bs64, [64, 1, 1]),
            (_launch_bs128_4_2, [128, 4, 2]),
            (_launch_no_block_size, [64, 1, 1]),
        ],
        ids=["explicit-1d", "explicit-3d", "inferred"],
    )
    def test_source_ir_contains_known_block_size(self, launch_fn, expected):
        source_ir = _get_source_ir(launch_fn, self.x)
        attr_str = f"known_block_size = array<i32: {expected[0]}, {expected[1]}, {expected[2]}>"
        assert attr_str in source_ir, f"expected '{attr_str}' in source IR, got:\n{source_ir}"

    def test_static_numeric_dims_are_inferred(self):
        source_ir = _get_source_ir(_launch_static_numeric_dims, self.x)
        assert "known_block_size = array<i32: 64, 1, 1>" in source_ir
        assert "known_block_size = array<i32: 32, 2, 1>" in source_ir

    def test_compiled_ir_has_max_flat_workgroup_size(self):
        compiled_ir = _get_compiled_ir(_launch_bs128_4_2, self.x)
        # The compiled IR should report max_flat_workgroup_size >= total_threads
        match = re.search(r"max_flat_workgroup_size\s*=\s*(\d+)", compiled_ir)
        assert match is not None, f"max_flat_workgroup_size not found in compiled IR:\n{compiled_ir}"
        max_wg = int(match.group(1))
        assert max_wg >= 1024, f"max_flat_workgroup_size={max_wg} < total_threads=1024"

    def test_dynamic_block_size_omits_exact_attribute(self):
        source_ir = _get_source_ir(_launch_dynamic_blocks, self.x, 64, 64)
        # Check for the attribute syntax, not just the substring (which may
        # appear in kernel names like "_kn_no_block_size_0").
        assert (
            "known_block_size = array<i32:" not in source_ir
        ), f"known_block_size attribute should not be inferred from a dynamic launch dimension:\n{source_ir}"

    def test_explicit_size_remains_contract_for_dynamic_block(self):
        source_ir = _get_source_ir(_launch_explicit_size_with_dynamic_block, self.x, 64)
        assert "known_block_size = array<i32: 64, 1, 1>" in source_ir

    def test_reused_pending_call_emits_one_specialization_per_launch(self):
        source_ir = _get_source_ir(_launch_pending_twice, self.x)
        assert source_ir.count("known_block_size = array<i32: 64, 1, 1>") == 1
        assert source_ir.count("known_block_size = array<i32: 128, 1, 1>") == 1

    def test_reused_pending_call_reuses_matching_specialization(self):
        source_ir = _get_source_ir(_launch_pending_same_size_twice, self.x)
        assert source_ir.count("known_block_size = array<i32: 64, 1, 1>") == 1
        assert source_ir.count("gpu.launch_func") == 2

    def test_reused_explicit_name_gets_unique_symbols(self):
        source_ir = _get_source_ir(_launch_named_pending_twice, self.x)
        assert "gpu.func @_kn_named_block_specialization(" in source_ir
        assert "gpu.func @_kn_named_block_specialization_1(" in source_ir

    def test_kernel_launches_successfully(self):
        """Ensure the kernel actually launches without hipErrorLaunchFailure."""
        _launch_bs64(self.x, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()  # would raise if launch failed


class TestKnownBlockSizeValidation:
    """Verify that invalid known_block_size values are rejected early."""

    def test_wrong_length_2(self):
        with pytest.raises(ValueError, match="exactly 3 elements"):

            @flyc.kernel(known_block_size=[256, 1])
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

    def test_none_is_accepted(self):
        """None enables launch-time inference and should not raise."""

        @flyc.kernel(known_block_size=None)
        def _ok(x: fx.Tensor):
            pass

    def test_static_numeric_elements_are_accepted(self):
        @flyc.kernel(known_block_size=[fx.Int32(64), fx.Int64(1), 1])
        def _ok(x: fx.Tensor):
            pass

        assert _ok._known_block_size == [64, 1, 1]


class TestKnownBlockSizeLaunchMismatch:
    """Verify that errors are raised for invalid block size at launch time."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = torch.zeros(64, device="cuda", dtype=torch.float32)

    def test_mismatch_raises(self):
        @flyc.kernel(known_block_size=[256, 1, 1])
        def _kn_256(x: fx.Tensor):
            pass

        @flyc.jit
        def _launch_wrong(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            _kn_256(x).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream)

        with pytest.raises(ValueError, match="differs from known_block_size"):
            _launch_wrong(self.x, stream=torch.cuda.current_stream())

    def test_static_block_above_256_is_inferred(self):
        @flyc.kernel
        def _kn_none(x: fx.Tensor):
            pass

        @flyc.jit
        def _launch_big(x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            _kn_none(x).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream)

        source_ir = _get_source_ir(_launch_big, self.x)
        assert "known_block_size = array<i32: 512, 1, 1>" in source_ir
