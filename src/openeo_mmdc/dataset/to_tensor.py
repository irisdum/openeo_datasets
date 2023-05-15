import logging.config
import random
from typing import Literal

import numpy as np
import torch
import xarray

from openeo_mmdc.dataset.dataclass import OneMod

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


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


def get_crop_idx(
    crop_size: int,
    crop_type: Literal["Center", "Random"],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """return the coordinate, width and height of the window loaded
    by the SITS depending od the value of the attribute
    self.crop_type

    Args:
        rows: int
        cols: int
    """
    if crop_type == "Random":
        return randomcropindex(rows, cols, crop_size, crop_size)

    return int(rows // 2 - crop_size), int(cols // 2 - crop_size)


def time_delta(date: np.ndarray, reference_date: np.datetime64 | None = None):
    if reference_date is None:
        reference_date = np.datetime64("2014-03-03", "D")
    duration = date - reference_date
    return duration


def randomcropindex(
    img_h: int, img_w: int, cropped_h: int, cropped_w: int
) -> tuple[int, int]:
    """
    Generate random numbers for window cropping of the patch (used in rasterio window)
    Args:
        img_h ():
        img_w ():
        cropped_h ():
        cropped_w ():

    Returns:

    """
    assert img_h >= cropped_h
    assert img_w >= cropped_w
    height = random.randint(0, img_h - cropped_h)
    width = random.randint(0, img_w - cropped_w)
    return height, width
