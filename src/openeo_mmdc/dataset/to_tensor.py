import logging.config
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray
from torchvision import transforms

from openeo_mmdc.constant.torch_dataloader import D_MODALITY
from openeo_mmdc.dataset.dataclass import ModTransform, OneMod, Stats
from openeo_mmdc.dataset.transform import Clip

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
    transform=None,
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
    sits = torch.Tensor(sits.values)
    if transform is not None:
        sits = transform(sits)
    return OneMod(sits, torch.Tensor(time))


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


def read_csv_stat(path_csv) -> Stats:
    assert path_csv.exists(), f"No file found at {path_csv}"
    df_stats = pd.read_csv(path_csv, sep=",", index_col=0)
    my_logger.info(df_stats)
    return Stats(
        median=df_stats.loc["med"].tolist(),
        qmin=df_stats.loc["qmin"].tolist(),
        qmax=df_stats.loc["qmax"].tolist(),
    )


def merge_stats_agera5(path_dir_csv, l_agera_mod) -> Stats:
    l_df = []
    for mod in l_agera_mod:
        path_file = Path(path_dir_csv).joinpath(
            f"dataset_{D_MODALITY[mod]}.csv"
        )
        l_df += [pd.read_csv(path_file, index_col=0)]
    df_stats = pd.concat(l_df, axis=1)
    return Stats(
        median=df_stats.loc["med"].tolist(),
        qmin=df_stats.loc["qmin"].tolist(),
        qmax=df_stats.loc["qmax"].tolist(),
    )


def load_transform_one_mod(
    path_dir_csv: str | None = None,
    mod: Literal["s2", "s1_asc", "s1_desc", "dem"]
    | list[
        Literal[
            "dew_temp",
            "prec",
            "sol_rad",
            "temp_max",
            "temp_mean",
            "temp_min",
            "val_press",
            "wind_speed",
        ]
    ] = "s2",
) -> None | torch.nn.Module:
    if path_dir_csv is not None:
        if isinstance(mod, str):
            path_csv = Path(path_dir_csv).joinpath(
                f"dataset_{D_MODALITY[mod]}.csv"
            )
            stats = read_csv_stat(path_csv)
            scale = tuple(
                [float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)]
            )
            return torch.nn.Sequential(
                Clip(qmin=stats.qmin, qmax=stats.qmax),
                transforms.Normalize(mean=stats.median, std=scale),
            )
        elif isinstance(mod, list):
            stats = merge_stats_agera5(
                path_dir_csv=path_dir_csv, l_agera_mod=mod
            )
            scale = tuple(
                [float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)]
            )
            return torch.nn.Sequential(
                Clip(qmin=stats.qmin, qmax=stats.qmax),
                transforms.Normalize(mean=stats.median, std=scale),
            )
        else:
            raise NotImplementedError
    else:
        return None


def load_all_transforms(
    path_dir_csv,
    modalities: list[Literal["s2", "s1_asc", "s1_desc", "dem", "agera5"]],
):
    all_transform = {}
    for mod in modalities:
        all_transform[mod] = load_transform_one_mod(
            path_dir_csv=path_dir_csv, mod=mod
        )
    return ModTransform(**all_transform)
