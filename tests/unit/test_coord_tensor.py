import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


def test_coord_tensor_slicing():
    @flyc.jit
    def test(i: fx.Int32, output: fx.Tensor):
        row_base = fx.make_int_tuple(0)
        row_layout = fx.make_layout((128, 64), (1, 0))
        coord_tensor = fx.Tensor(fx.make_view(row_base, row_layout))
        ct0 = coord_tensor[6, None]
        ct1 = coord_tensor[i, None]
        assert ct0[0].get_static_leaf_int == 6, f"expected static slice to be 6, got {ct0[0].get_static_leaf_int}"
        output[0] = fx.get_scalar(ct0[0])
        output[1] = fx.get_scalar(ct1[0])

    output = torch.zeros(2, dtype=torch.int32, device="cpu", requires_grad=False)
    test(6, output)
    assert output[0].item() == 6, f"expected 6 for static slice, got {output[0].item()}"
    assert output[1].item() == 6, f"expected 6 for dynamic slice, got {output[1].item()}"
