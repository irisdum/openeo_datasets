import pytest

from openeo_mmdc.dataset.dataclass import MMDC_MAXLEN
from openeo_mmdc.dataset.load_tensor import (
    create_mmcd_tensor_df,
    load_mmdc_sits,
)

PATH_DATASET_PT = "/home/ad/dumeuri/scratch/datasets/MMDC_OE/PT_FORMAT/val"


@pytest.mark.hal
def test_load_mmdc_sits():
    df = create_mmcd_tensor_df(path_directory=PATH_DATASET_PT,
                               s2_tile=["30TYR"],
                               modalities=["s2"])
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
    assert len(
        out_mmdc_item.s2.sits.shape) == 4, f"len {out_mmdc_item.s2.sits.shape}"
    assert out_mmdc_item.s2.doy.shape[0] == 20
    assert out_mmdc_item.s2.sits.shape[1] == 20


@pytest.mark.hal
def test_create_mmcd_tensor_df():
    df = create_mmcd_tensor_df(
        path_directory=PATH_DATASET_PT,
        s2_tile=["31TEK", "34UDB", "30TYR"],
        modalities=["s2"],
    )
    assert len(df) > 0
    print(len(df))
    assert len(df) == 90
