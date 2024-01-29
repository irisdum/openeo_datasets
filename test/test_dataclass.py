import torch

from openeo_mmdc.dataset.dataclass import MaskMod, OneMod


def create_one_mode() -> OneMod:
    t, c, h, w = 15, 10, 16, 16
    sits = torch.randn(c, t, h, w)
    doy = torch.arange(0, t)
    mask_mod = MaskMod()
    return OneMod(sits=sits, doy=doy, mask=mask_mod)


def test_apply_padding():
    ex = create_one_mode()
    ex = ex.apply_padding(max_len=30, allow_padd=True)
    assert ex.sits.shape[0] == 30
    assert ex.mask.padd_mask.shape[0] == 30
    assert ex.mask.padd_mask[-1] == 1
    assert ex.doy.shape[0] == 30
