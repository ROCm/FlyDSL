# SPDX-License-Identifier: Apache-2.0
"""Check the gfx950 cache layout with CPU tensors and native ATOM decoders."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("atom")

if not torch.cuda.is_available():
    pytest.skip("native ATOM decoder imports require an active ROCm device", allow_module_level=True)
if not getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith("gfx950"):
    pytest.skip("this fixture checks the gfx950 scale-row layout", allow_module_level=True)

from kernels.monokernel.dsv4.atom_layer import AtomLayerModule  # noqa: E402


def test_completed_row_uses_physical_page_and_transposed_scale_row():
    # Position 383 commits logical compressed row 95: page 1, local row 31.
    # Native gfx950 puts this row's scales at (31 % 16) * 4 + 31 // 16 = 61.
    main_storage = torch.zeros(145, 512, dtype=torch.uint8)
    rope_storage = torch.zeros(145, 64, dtype=torch.bfloat16)
    main, rope = main_storage[5:], rope_storage[5:]
    main[101, :448] = torch.ones(448).to(torch.float8_e4m3fn).view(torch.uint8)
    main[101, 448:462] = 127
    rope[101] = 2
    data = torch.zeros(2, 1, 4, 64, 16, dtype=torch.uint8)
    scales = torch.full((2, 1, 4, 64), 120, dtype=torch.uint8)
    data[1, 0, :, 31] = 0x21  # Alternating E2M1 values 0.5 and 1.
    scales[1, 0, :, 61] = torch.tensor([125, 126, 127, 128], dtype=torch.uint8)
    indexer = SimpleNamespace(_indexer_fp4=True, kv_cache=data, kv_scale=scales)
    attn = SimpleNamespace(
        compress_ratio=4, unified_kv=main, unified_kv_rope=rope, indexer=indexer, named_modules=lambda: []
    )
    fixture = object.__new__(AtomLayerModule)
    fixture.block = SimpleNamespace(attn=attn)
    md = SimpleNamespace(state_slot_out_cpu=[0], envelope_rows=70, block_tables=torch.tensor([[0, 1]]))
    ctx = SimpleNamespace(positions=torch.tensor([381, 382, 383, 384]))
    values, regions = fixture.cache_observation(md, ctx)
    assert set(values) == {"main_compressed.95", "index_compressed.95"}
    torch.testing.assert_close(
        values["main_compressed.95"], torch.cat((torch.ones(448), torch.full((64,), 2.0))).bfloat16()
    )
    expected = torch.tensor([0.5, 1.0]).repeat(16).expand(4, -1)
    expected = expected * torch.tensor([0.25, 0.5, 1.0, 2.0])[:, None]
    torch.testing.assert_close(values["index_compressed.95"], expected, rtol=0, atol=0)

    # Numerical regions must cover only that row's payload, including view
    # offsets. A corrupted scale from any other row or padding must still fail.
    def byte_region(t):
        assert t.is_contiguous()
        return (t.untyped_storage().data_ptr(), t.storage_offset() * t.element_size(), t.numel() * t.element_size())

    wanted = [main[101, :462], rope[101]]
    for group in range(4):
        wanted += [data[1, 0, group, 31], scales[1, 0, group, 61:62]]
    assert [byte_region(t) for t in regions] == [byte_region(t) for t in wanted]
    main[101, 449] = 126
    with pytest.raises(AssertionError, match="duplicated E8M0"):
        fixture.cache_observation(md, ctx)
