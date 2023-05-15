import pytest

from openeo_mmdc.dataset.to_tensor import load_transform_one_mod


@pytest.mark.local
def test_load_transform_one_mod():
    out = load_transform_one_mod(
        path_dir_csv="/media/dumeuri/DATA/Data/Iris/MMDC_OE", mod="s2"
    )
    assert out is not None
