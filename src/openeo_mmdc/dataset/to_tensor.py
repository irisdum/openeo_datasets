import random
from typing import Literal

import torch
import xarray

from openeo_mmdc.dataset.dataclass import OneMod
from openeo_mmdc.dataset.utils import get_crop_idx, my_logger, time_delta


def from_dataset2tensor(
    dataset: xarray.Dataset,
    max_len: int = 10,
    crop_size=64,
    crop_type: Literal["Center", "Random"] = "Center",
) -> OneMod:
    d_s = dataset.sizes

    temp_idx = sorted(random.sample([i for i in range(d_s["t"])], max_len))
    dataset = dataset.isel({"t": temp_idx})
    time_info = xarray.apply_ufunc(time_delta, dataset.coords["t"])
    time = time_info.values.astype(dtype="timedelta64[D]")
    time = time.astype("int32")
    sits = dataset.to_array()
    row, cols = sits.shape[-2], sits.shape[-1]
    x, y = get_crop_idx(
        rows=row, cols=cols, crop_size=crop_size, crop_type=crop_type
    )
    my_logger.debug(sits.shape)
    sits = sits[:, :, x : x + crop_size, y : y + crop_size]
    return OneMod(torch.Tensor(sits.values), torch.Tensor(time))
