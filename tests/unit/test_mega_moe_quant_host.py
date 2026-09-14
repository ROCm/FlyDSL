# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import gc
import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch

import kernels.mega_moe.mega_moe_stage2 as stage2_module
from kernels.comm.flydsl_dispatch_combine_intranode_op import FlyDSLDispatchCombineConfig
from kernels.mega_moe import (
    convert_aiter_lqq_to_megamoe as exported_convert_aiter_lqq_to_megamoe,
)
from kernels.mega_moe.mega_moe import (
    MegaMoEV2,
    _combine_launch_geometry,
    _dispatch_quant_config,
    _shared_runtime_signature,
)
from kernels.mega_moe.mega_moe_config import Stage2Config
from kernels.mega_moe.quant import (
    convert_aiter_lqq_to_megamoe,
    repack_megamoe_lqq_for_int8_loader,
)


@pytest.mark.parametrize(
    "quant,expected",
    [
        ("a8w4", (torch.float8_e4m3fn, 8, 1)),
        ("a8w4smooth", (torch.int8, 1, 4)),
        ("w8a8smooth", (torch.int8, 1, 4)),
    ],
)
def test_quant_mode_dispatch_contract(quant, expected):
    assert _dispatch_quant_config(quant, 256) == expected


def test_unknown_quant_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported quant"):
        _dispatch_quant_config("int8", 256)


def test_mxfp4_transport_is_w8a8smooth_prefill_only():
    assert _dispatch_quant_config("w8a8smooth", 3584, "mxfp4") == (
        torch.float4_e2m1fn_x2,
        112,
        1,
    )
    for quant in ("a8w4", "a8w4smooth"):
        with pytest.raises(ValueError, match="only supported by w8a8smooth"):
            _dispatch_quant_config(quant, 3584, "mxfp4")


@pytest.mark.parametrize(
    "quant,mtpr,expected",
    [
        ("a8w4smooth", 1, (64, 4)),
        ("a8w4smooth", 4, (32, 4)),
        ("a8w4smooth", 8, (64, 4)),
        ("a8w4smooth", 16, (None, None)),
        ("w8a8smooth", 4, (None, None)),
        ("a8w4", 4, (None, None)),
    ],
)
def test_small_a8w4_combine_geometry(quant, mtpr, expected):
    assert _combine_launch_geometry(quant, mtpr) == expected


def test_int8_combine_config_contract():
    cfg = FlyDSLDispatchCombineConfig(
        rank=0,
        world_size=1,
        hidden_dim=256,
        max_num_inp_token_per_rank=16,
        num_experts_per_rank=8,
        num_experts_per_token=2,
        dispatch_dtype=torch.int8,
        combine_dtype=torch.bfloat16,
        scale_dim=1,
        scale_type_size=4,
    )
    assert cfg.dispatch_dtype == torch.int8
    assert cfg.combine_dtype == torch.bfloat16
    assert cfg.scale_bytes == 4


def test_a8w4_constructor_contract_is_unchanged():
    params = inspect.signature(MegaMoEV2.__init__).parameters
    required = (
        "rank",
        "world_size",
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "quant",
        "w1",
        "w1_scale",
        "w2",
        "w2_scale",
        "max_tok_per_rank",
    )
    assert all(params[name].default is inspect.Parameter.empty for name in required)
    assert params["mega_scheme"].default == "fixedslot"
    assert params["swiglu_limit"].default == 0.0
    for name in (
        "w1_lqq_scale",
        "w1_lqq_zero",
        "w2_lqq_scale",
        "w2_lqq_zero",
        "fc1_smooth_scale",
        "fc2_smooth_scale",
    ):
        assert params[name].default is None
    assert params["weight_format"].default == "megamoe"
    assert params["dispatch_quant"].default is None
    assert params["shared_instance"].default is None


def _m13_comb_config(*, mtpr=128):
    return FlyDSLDispatchCombineConfig(
        rank=0,
        world_size=8,
        hidden_dim=3584,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=48,
        num_experts_per_token=8,
        combine_dtype=torch.bfloat16,
        dispatch_dtype=torch.float8_e4m3fn,
        scale_dim=112,
        scale_type_size=1,
        enable_std_moe=False,
        enable_group_major=True,
        gm_unit_size=32,
        gm_scheme="fixedslot",
        gm_compact=False,
        max_total_recv_tokens=8,
    )


def _host_only_owner(*, mtpr=128):
    config = _m13_comb_config(mtpr=mtpr)
    owner = object.__new__(MegaMoEV2)
    owner._mega_moe_shared_runtime_signature = _shared_runtime_signature(
        config,
        quant="a8w4",
        inter_dim=1280,
        dispatch_quant=None,
        weight_format="megamoe",
        swiglu_limit=0.0,
    )
    owner._is_int8_smooth = False
    owner.comb_op = object()
    owner._s1_epoch_parity = torch.zeros(1)
    owner._s1_dispatch_workspace = {"entry_count": torch.zeros(1)}
    owner._s1_out = torch.zeros(2)
    owner._s1_osd = torch.zeros(3, dtype=torch.uint8)
    owner._g2_combine_placeholder = torch.zeros(4)
    return owner


def _make_shared_native(owner, *, mtpr=128, value=0.0):
    weight = torch.tensor([value])
    return MegaMoEV2(
        rank=0,
        world_size=8,
        model_dim=3584,
        inter_dim=1280,
        experts=384,
        topk=8,
        quant="a8w4",
        w1=weight,
        w1_scale=weight,
        w2=weight,
        w2_scale=weight,
        max_tok_per_rank=mtpr,
        shared_instance=owner,
    )


def test_shared_instance_rejects_an_uninitialized_instance():
    with pytest.raises(ValueError, match="not fully initialized"):
        _make_shared_native(object.__new__(MegaMoEV2))


def test_failed_owner_construction_is_not_shareable(monkeypatch):
    owner = object.__new__(MegaMoEV2)
    weight = torch.tensor([0.0])

    def fail_runtime_construction(_config):
        raise RuntimeError("injected communication allocation failure")

    monkeypatch.setattr(
        "kernels.mega_moe.mega_moe.FlyDSLDispatchCombineIntraNodeOp",
        fail_runtime_construction,
    )
    with pytest.raises(RuntimeError, match="injected communication"):
        MegaMoEV2.__init__(
            owner,
            rank=0,
            world_size=8,
            model_dim=3584,
            inter_dim=1280,
            experts=384,
            topk=8,
            quant="a8w4",
            w1=weight,
            w1_scale=weight,
            w2=weight,
            w2_scale=weight,
            max_tok_per_rank=128,
        )

    assert not hasattr(owner, "_mega_moe_shared_runtime_signature")
    with pytest.raises(ValueError, match="not fully initialized"):
        _make_shared_native(owner)


def test_shared_instance_rejects_the_wrong_type():
    with pytest.raises(TypeError, match="must be a MegaMoEV2"):
        _make_shared_native(object())


def test_shared_instance_rejects_a_config_mismatch():
    with pytest.raises(ValueError, match="max_num_inp_token_per_rank"):
        _make_shared_native(_host_only_owner(mtpr=128), mtpr=64)


def test_76_instances_reuse_the_complete_runtime_workspace():
    owner = _host_only_owner()
    instances = [owner]
    for index in range(1, 76):
        # Frameworks may naturally pass the preceding layer rather than keep
        # the first owner separately.  Nested sharing must still resolve to
        # the single root runtime.
        instances.append(_make_shared_native(instances[-1], value=index))

    assert all(instance.comb_op is owner.comb_op for instance in instances)
    assert all(instance._shared_runtime is owner for instance in instances[1:])
    assert all(instance._s1_epoch_parity is owner._s1_epoch_parity for instance in instances)
    assert all(instance._s1_dispatch_workspace is owner._s1_dispatch_workspace for instance in instances)
    for name in ("_s1_out", "_s1_osd", "_g2_combine_placeholder"):
        assert all(getattr(instance, name) is getattr(owner, name) for instance in instances)
    assert [instance._s1_w1.item() for instance in instances[1:]] == list(map(float, range(1, 76)))
    assert not hasattr(owner, "_s1_w1")


def test_shared_instance_keeps_the_runtime_owner_alive():
    owner = _host_only_owner()
    owner_ref = weakref.ref(owner)
    instance = _make_shared_native(owner)

    del owner
    gc.collect()

    assert owner_ref() is instance._shared_runtime
    assert instance.comb_op is instance._shared_runtime.comb_op


def test_smooth_forward_does_not_require_live_tokens_to_equal_capacity():
    moe = object.__new__(MegaMoEV2)
    moe.mtpr = 8
    moe.quant = "a8w4smooth"
    moe._is_int8_smooth = True
    sentinel = object()
    moe._forward_int8 = lambda *args, **kwargs: sentinel

    x = torch.empty((3, 16), dtype=torch.bfloat16)
    weights = torch.empty((3, 2), dtype=torch.float32)
    expert_ids = torch.empty((3, 2), dtype=torch.int32)

    assert moe.forward(x, weights, expert_ids) is sentinel


def test_int8_stage2_rejects_native_n_major_before_runtime_access():
    moe = object.__new__(MegaMoEV2)
    moe._active_config = SimpleNamespace(
        stage2=Stage2Config(
            block_m=32,
            block_n=128,
            persist=True,
            persist_cu=240,
            use_nt=False,
            persist_n_major=True,
        )
    )
    with pytest.raises(ValueError, match="only supported by native A8W4 Stage2"):
        moe._run_int8_stage2(None, None, None, 0, None, False)


def test_n_major_stage2_requires_persistent_mode(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(AssertionError, match="persist_n_major=True requires persist=True"):
        stage2_module.compile_mega_moe_stage2(
            model_dim=3584,
            inter_dim=1280,
            experts=48,
            topk=8,
            rank=0,
            npes=8,
            max_tok=8192,
            persist=False,
            persist_n_major=True,
        )


def test_lqq_conversion_shape_and_layout_formula():
    assert exported_convert_aiter_lqq_to_megamoe is convert_aiter_lqq_to_megamoe
    experts, rows, k_dim = 1, 16, 256
    u4 = (torch.arange(experts * rows * k_dim, dtype=torch.int64) % 16).to(torch.uint8)
    u4 = u4.view(experts, rows, k_dim)
    scale = torch.arange(experts * rows * (k_dim // 64), dtype=torch.uint8).view(experts, rows, k_dim // 64)
    zero = (255 - scale).to(torch.uint8)

    weight, packed_scale, packed_zero = convert_aiter_lqq_to_megamoe(u4, scale, zero)

    assert weight.dtype == torch.int8
    assert weight.shape == (experts * rows * k_dim // 2,)
    assert packed_scale.shape == packed_zero.shape == (experts, rows // 16, k_dim // 256, 16)
    assert packed_scale.dtype == packed_zero.dtype == torch.int32

    # Independent scalar reference for the (16,16) preshuffle, K64 interleave,
    # and low/high-nibble byte packing.
    shuffled = []
    for k_block in range(k_dim // 32):
        for row in range(rows):
            shuffled.extend(u4[0, row, k_block * 32 : (k_block + 1) * 32].tolist())
    shuffled = torch.tensor(shuffled, dtype=torch.uint8).view(rows, k_dim)
    interleaved = torch.empty_like(shuffled)
    for row in range(rows):
        for chunk in range(k_dim // 128):
            source = shuffled[row, chunk * 128 : (chunk + 1) * 128]
            interleaved[row, chunk * 128 : (chunk + 1) * 128 : 2] = source[:64]
            interleaved[row, chunk * 128 + 1 : (chunk + 1) * 128 : 2] = source[64:]
    expected_weight = interleaved.reshape(-1, 2)[:, 0] | (interleaved.reshape(-1, 2)[:, 1] << 4)
    assert torch.equal(weight.view(torch.uint8), expected_weight)

    for row in range(rows):
        expected_scale = sum(int(scale[0, row, group]) << (8 * group) for group in range(4))
        expected_zero = sum(int(zero[0, row, group]) << (8 * group) for group in range(4))
        expected_scale = torch.tensor(expected_scale & 0xFFFFFFFF, dtype=torch.uint32).view(torch.int32)
        expected_zero = torch.tensor(expected_zero & 0xFFFFFFFF, dtype=torch.uint32).view(torch.int32)
        assert packed_scale[0, 0, 0, row] == expected_scale
        assert packed_zero[0, 0, 0, row] == expected_zero


def test_lqq_conversion_rejects_invalid_shape():
    u4 = torch.zeros((1, 16, 128), dtype=torch.uint8)
    qparam = torch.zeros((1, 16, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="K % 256"):
        convert_aiter_lqq_to_megamoe(u4, qparam, qparam)


def test_legacy_lqq_repack_matches_direct_k64_pairs():
    rows = k_dim = 256
    u4 = torch.randint(0, 16, (1, rows, k_dim), dtype=torch.uint8)
    qparam = torch.ones((1, rows, k_dim // 64), dtype=torch.uint8)
    legacy, _, _ = convert_aiter_lqq_to_megamoe(u4, qparam, qparam)
    repacked = repack_megamoe_lqq_for_int8_loader(legacy, rows, k_dim)

    full_layout = (
        u4.view(1, rows // 16, 16, k_dim // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(1, rows // 16, k_dim // 64, 4, 16, 16)
    )
    expected = (full_layout[:, :, 0::2] | (full_layout[:, :, 1::2] << 4)).contiguous()
    assert torch.equal(repacked.view(torch.uint8), expected.view(-1))
