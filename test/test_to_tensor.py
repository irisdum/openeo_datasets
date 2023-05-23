from pathlib import Path

import pytest
import torch
import xarray

from openeo_mmdc.dataset.to_tensor import (
    from_dataset2tensor,
    load_transform_one_mod,
)


@pytest.mark.local
def test_load_transform_one_mod():
    out = load_transform_one_mod(
        path_dir_csv="/media/dumeuri/DATA/Data/Iris/MMDC_OE", mod="s2"
    )
    assert out is not None


@pytest.mark.local
def test_from_dataset2tensor(crop_size=64, max_len=20):
    PATH_DIR = (
        "/media/dumeuri/DATA/Data/Iris/MMDC_OE/31TEK/2017/Sentinel2_31TEK_2017"
    )
    IM = "openEO_0_clip.nc"
    PATH_SITS = Path(PATH_DIR).joinpath(IM)
    dataset = xarray.open_dataset(PATH_SITS, decode_cf=False)
    band = ["B02", "B03", "B04"]
    out_mod = from_dataset2tensor(
        dataset=dataset,
        max_len=max_len,
        crop_size=crop_size,
        crop_type="Center",
        transform=None,
        band_cld=["CLM", "SCL"],
        load_variable=band,
    )
    print(out_mod.sits.shape)
    print(torch.unique(out_mod.mask.mask_slc))
    print(out_mod.doy)
    print(
        torch.unique(
            out_mod.mask.mask_cld[~torch.isnan(out_mod.mask.mask_cld)]
        )
    )
    assert out_mod.sits.shape[-1] == crop_size
    assert out_mod.mask.mask_cld.shape[-1] == out_mod.sits.shape[-1]
    assert out_mod.sits.shape[1] == max_len
    assert out_mod.sits.shape[0] == len(band)
