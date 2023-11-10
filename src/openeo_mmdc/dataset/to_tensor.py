import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray
from torch import Tensor
from xarray import DataArray, Dataset

from openeo_mmdc.constant.torch_dataloader import D_MODALITY
from openeo_mmdc.dataset.dataclass import (
    MaskMod,
    ModTransform,
    OneMod,
    OneTransform,
    Stats,
)
from openeo_mmdc.dataset.transform import Clip, S2Normalize

my_logger = logging.getLogger(__name__)


def light_from_dataset2tensor(
    dataset: xarray.Dataset,
    band_cld: list | None = None,
    load_variable: list | None = None,
):
    time_info = xarray.apply_ufunc(time_delta, dataset.coords["t"])
    time = time_info.values.astype(dtype="timedelta64[D]")
    time = time.astype("int32")
    if load_variable is not None:
        spectral_dataset = dataset[load_variable]
        if band_cld is not None:
            cld_dataset = dataset[band_cld]
    else:
        spectral_dataset = dataset
    sits = spectral_dataset.to_array()
    sits = torch.Tensor(sits.values)
    if band_cld is not None:
        nan_mask = torch.isnan(torch.sum(sits, dim=0, keepdim=False))
        cld_mask = torch.Tensor(cld_dataset[["CLM"]].to_array().values)
        mask_sits = MaskMod(
            mask_cld=cld_mask,
            mask_nan=nan_mask,
            mask_slc=torch.Tensor(cld_dataset[["SCL"]].to_array().values),
        )
        # print(f"mask cld {cld_mask[0,:,0,0]}")
    else:
        my_logger.debug("No cld mask applied")
        mask_sits = MaskMod()
    return OneMod(sits, torch.Tensor(time), mask=mask_sits)


def from_dataset2tensor(
    dataset: xarray.Dataset,
    max_len: int | None = 10,
    crop_size=64,
    crop_type: Literal["Center", "Random"] = "Center",
    transform=None,
    band_cld: list | None = None,
    load_variable: list | None = None,
    seed: int | None = None,
) -> OneMod:
    d_s = dataset.sizes
    if seed is not None:
        random.seed(seed)
    if max_len is not None:
        temp_idx = sorted(random.sample([i for i in range(d_s["t"])], max_len))
        dataset = dataset.isel({"t": temp_idx})
    time_info = xarray.apply_ufunc(time_delta, dataset.coords["t"])
    time = time_info.values.astype(dtype="timedelta64[D]")
    time = time.astype("int32")

    if load_variable is not None:
        my_logger.debug(load_variable)
        spectral_dataset = dataset[list(load_variable)]
        my_logger.debug(band_cld)
        if band_cld is not None:
            cld_dataset = dataset[band_cld]

    else:
        spectral_dataset = dataset
    sits = spectral_dataset.to_array()

    row, cols = sits.shape[-2], sits.shape[-1]
    x, y = get_crop_idx(
        rows=row, cols=cols, crop_size=crop_size, crop_type=crop_type
    )
    my_logger.debug(sits.shape)
    sits = sits[:, :, x : x + crop_size, y : y + crop_size]
    sits = torch.Tensor(sits.values)
    my_logger.debug(band_cld)
    if band_cld is not None:
        # nan_mask = torch.isnan(torch.sum(sits, dim=0, keepdim=False))

        cld_mask = crop_dataset(cld_dataset[["CLM"]], x, y, crop_size)
        nan_mask = crop_dataset(cld_dataset[["SCL"]], x, y, crop_size) == 0

        mask_sits = MaskMod(
            mask_cld=cld_mask,
            mask_nan=nan_mask,
            mask_slc=crop_dataset(cld_dataset[["SCL"]], x, y, crop_size),
        )
        # print(f"mask cld {cld_mask[0,:,0,0]}")
    else:
        my_logger.info("No cld mask applied")
        mask_sits = MaskMod()
    if transform is not None:
        my_logger.debug("apply transform")
        sits = transform(sits)
        assert torch.count_nonzero(torch.isnan(sits)) == 0, "Nan input"
    return OneMod(sits, torch.Tensor(time), mask=mask_sits)


def crop_tensor(
    tensor: DataArray | Tensor, x, y, crop_size
) -> DataArray | Tensor:
    """
    Args:
        tensor (): spatial dimension need to be the last two dimension of the tensor
        x ():
        y ():
        crop_size ():

    Returns:

    """
    return tensor[..., x : x + crop_size, y : y + crop_size]


def crop_dataset(dataset: Dataset, x, y, crop_size) -> Tensor:
    array = dataset.to_array()
    return torch.Tensor(crop_tensor(array, x, y, crop_size).values)


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

    return int(rows - crop_size) // 2, int(cols - crop_size) // 2


def time_delta(
    int_date: np.ndarray,
    reference_date: np.datetime64 | None = None,
    scale: float | None = None,
):
    date = np.datetime64("1970-01-01") + int_date.astype(
        int
    )  # TODO set in constant file
    if reference_date is None:
        reference_date = np.datetime64("2014-03-03", "D")
    duration = date - reference_date
    if scale is not None:
        return duration / scale
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
) -> [None | torch.nn.Module, Stats]:
    if path_dir_csv is not None:
        if isinstance(mod, str):
            path_csv = Path(path_dir_csv).joinpath(
                f"dataset_{mod}.csv"
            )  # TODO change that for weather meteo
            stats = read_csv_stat(path_csv)
            scale = tuple(
                [float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)]
            )
            if mod == "s2":
                return OneTransform(
                    torch.nn.Sequential(
                        Clip(
                            qmin=stats.qmin, qmax=stats.qmax, s2_partial=False
                        ),
                        S2Normalize(
                            med=stats.median, scale=scale, s2_partial=False
                        ),
                    ),
                    stats,
                )
            return OneTransform(
                torch.nn.Sequential(
                    Clip(qmin=stats.qmin, qmax=stats.qmax),
                    S2Normalize(med=stats.median, scale=scale),
                ),
                stats,
            )
        elif isinstance(mod, list):
            stats = merge_stats_agera5(
                path_dir_csv=path_dir_csv, l_agera_mod=mod
            )
            scale = tuple(
                [float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)]
            )
            return OneTransform(
                torch.nn.Sequential(
                    Clip(qmin=stats.qmin, qmax=stats.qmax),
                    S2Normalize(med=stats.median, scale=scale),
                ),
                stats,
            )
        else:
            raise NotImplementedError
    else:
        return None


def load_all_transforms(
    path_dir_csv,
    modalities: list[Literal["s2", "s1_asc", "s1_desc", "dem", "agera5"]],
) -> ModTransform:
    all_transform = {}
    for mod in modalities:
        all_transform[mod] = load_transform_one_mod(
            path_dir_csv=path_dir_csv, mod=mod
        )
    return ModTransform(**all_transform)
