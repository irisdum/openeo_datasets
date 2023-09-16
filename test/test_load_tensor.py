import pytest
import torch

from openeo_mmdc.dataset.load_tensor import crop_spat_temp


@pytest.mark.parametrize("padding_val", [(0), (10)])
def test_crop_spat_temp(padding_val):
    tensor = torch.randn(10, 30, 64, 64)
    list_t = [i for i in range(20)]
    out_t = crop_spat_temp(tensor, 0, 0, 32, list_t, padding_val)
    assert out_t.shape[1] == len(list_t) + padding_val
    if padding_val > 0:
        assert out_t[0, -padding_val, 0, 0] == 0
    out = crop_spat_temp(
        torch.ones(tensor.shape), 0, 0, 32, list_t, padding_val
    )
    if padding_val > 0:
        assert out[0, -padding_val, 0, 0] == 0
    print(out[0, :, 0, 0])
