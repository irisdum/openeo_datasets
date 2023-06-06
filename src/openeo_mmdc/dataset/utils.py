import logging.config
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
import torch.nn
import xarray
from torch import Tensor
from xarray import Dataset

from openeo_mmdc.constant.torch_dataloader import D_MODALITY, FORMAT_SITS
from openeo_mmdc.dataset.dataclass import MMDCDF
from openeo_mmdc.dataset.to_tensor import from_dataset2tensor

logging.config.dictConfig({"version": 1})
my_logger = logging.getLogger(__name__)


def load_mmdc_path(
    path_dir: str,
    s2_tile: list[str],
    modality: Literal[
        "s2",
        "S1_asc",
        "s1_desc",
        "dem",
        "dew_temp",
        "prec",
        "sol_rad",
        "temp_max",
        "temp_mean",
        "temp_min",
        "val_press",
        "wind_speed",
    ],
) -> pd.DataFrame:
    assert Path(path_dir).exists(), f"{path_dir} not found"
    list_available_sits = [
        p for p in Path(path_dir).rglob(
            f"*/*{D_MODALITY[modality]}*/**/*_{FORMAT_SITS}")
    ]
    my_logger.debug(list_available_sits[0])
    my_logger.debug(f"*/*{D_MODALITY[modality]}*/**/*_{FORMAT_SITS}")
    available_patch = list({elem.name for elem in list_available_sits})
    my_logger.debug(f"available patch{available_patch}")
    l_df = []
    for tile in s2_tile:
        for patch_id in sorted(available_patch):
            pattern = f"{tile}/**/*{D_MODALITY[modality]}*/**/{patch_id}"
            my_logger.debug(f"path dir{path_dir} patch id {patch_id}")
            l_s2 = [p for p in Path(path_dir).rglob(pattern)]
            id_series = {
                "mod": modality,
                "patch_id": patch_id,
                "s2_tile": tile,
                "sits_path": l_s2,
            }
            assert l_s2, f"No image found at {pattern}"
            l_df += [pd.DataFrame([id_series])]
    my_logger.debug(l_df)
    final_df = pd.concat(l_df, ignore_index=True)
    return final_df


def build_dataframe(path_dir, l_modality: list,
                    s2_tile: list[str]) -> pd.DataFrame:
    l_mod_df = []
    for mod in l_modality:
        l_mod_df += [load_mmdc_path(path_dir, modality=mod, s2_tile=s2_tile)]
    return pd.concat(l_mod_df)


def build_dataset_info(
    path_dir: str,
    l_tile_s2: list[str],
    list_modalities: list[Literal[
        "s2",
        "s1_asc",
        "s1_desc",
        "dem",
        "dew_temp",
        "prec",
        "sol_rad",
        "temp_max",
        "temp_mean",
        "temp_min",
        "val_press",
        "wind_speed",
    ]],
) -> MMDCDF:  # TODO deal with AGERA5 data
    d_mod = {}
    for mod in list_modalities:
        d_mod[mod] = load_mmdc_path(path_dir=path_dir,
                                    modality=mod,
                                    s2_tile=l_tile_s2)
    return MMDCDF(**d_mod)


def load_one_mod(sits_pattern):
    return xarray.open_mfdataset(sits_pattern)


def load_item_dataset_modality(
    mod_df: pd.DataFrame,
    item: int,
    drop_variable: Hashable | Iterable[Hashable] = None,
    load_variables: list = None,
    s2_max_ccp: float | None = None,
) -> Dataset:
    my_logger.debug(f"item{item}")
    item_series = mod_df.iloc[item]
    tile = item_series["s2_tile"]
    modality = item_series["mod"]
    patch_id = item_series["patch_id"]
    pattern = f"{tile}/**/*{D_MODALITY[modality]}*/**/{patch_id}"
    my_logger.debug(f"pattern{pattern}")
    # global_pattern = os.path.join(path_dir, pattern)
    # print(item_series["sits_path"])
    path_im = item_series["sits_path"]
    my_logger.debug(f"we are loading {path_im}")
    dataset = xarray.open_mfdataset(path_im,
                                    combine="nested",
                                    decode_cf=False,
                                    chunks="auto")
    my_logger.debug(f"load var{load_variables}")
    if drop_variable is not None:
        my_logger.debug(f"drop_var {drop_variable}")
        dataset = dataset.drop_vars(names=drop_variable)
    if load_variables:
        #        dataset = order_dataset_vars(dataset, list_vars_order=load_variables)
        my_logger.debug(dataset)
        my_logger.debug(load_variables)
        dataset = dataset[list(load_variables)]

    my_logger.debug(f"after {list(dataset.data_vars)}")
    if s2_max_ccp is not None:
        my_logger.debug("max cc")
        max_pix_cc = dataset.sizes["x"] * s2_max_ccp
        ccp = dataset[["CLM"]].sum(dim=["x", "y"])
        ccp = ccp.compute()
        dataset = dataset.sel(
            t=ccp.where(ccp["CLM"] < max_pix_cc, drop=True)["t"])
    return dataset


def order_dataset_vars(dataset, list_vars_order=None):

    if list_vars_order is None:
        sorted_vars = sorted(dataset.data_vars)

    else:
        sorted_vars = list_vars_order

    return dataset[sorted_vars]


def merge_agera5_datasets(
    l_agera5_df: list[pd.DataFrame],
    item: int,
    max_len: int = 10,
    crop_size=64,
    crop_type: Literal["Center", "Random"] = "Center",
    transform: None | torch.nn.Module = None,
    seed=None,
) -> (Tensor, Tensor):
    l_dataset_agera5 = [
        load_item_dataset_modality(mod_df, item) for mod_df in l_agera5_df
    ]
    tot_array = xarray.merge(l_dataset_agera5)  # c,t,h,w
    return from_dataset2tensor(
        tot_array,
        max_len=max_len,
        crop_size=crop_size,
        crop_type=crop_type,
        transform=transform,
        seed=seed,
    )
