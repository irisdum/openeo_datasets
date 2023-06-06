import pytest

from openeo_mmdc.dataset.dataclass import MMDC_MAXLEN
from openeo_mmdc.dataset.load_tensor import (
    create_mmcd_tensor_df,
    load_mmdc_sits,
)

PATH_DATASET_PT = "/home/ad/dumeuri/scratch/datasets/MMDC_OE/val/PT_FORMAT"


@pytest.marker.hal
def test_load_mmdc_sits():
    df = create_mmcd_tensor_df(
        path_directory=PATH_DATASET_PT, s2_tile=["3OTYR"], modalities=["s2"]
    )
    item = 0
    out_mmdc_item = load_mmdc_sits(
        c_mmdc_df=df,
        item=item,
        crop_size=64,
        crop_type="Center",
        max_len=MMDC_MAXLEN(20, 20, 20, 20),
        opt="s2",
        all_transform=None,
        seed=1,
    )
    assert len(out_mmdc_item.s2.sits.shape) == 4
    assert out_mmdc_item.s2.doy.shape == 20


@pytest.marker.hal
def test_create_mmcd_tensor_df():
    df = create_mmcd_tensor_df(
        path_directory=PATH_DATASET_PT,
        s2_tile=["3OTYR"],
        modalities=["s2", "s1_asc", "s1_desc", "dem", "agera5"],
    )
    assert len(df) > 0
