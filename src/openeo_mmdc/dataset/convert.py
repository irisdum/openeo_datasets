import logging
from typing import Literal

import torch
import xarray

from openeo_mmdc.constant.torch_dataloader import (
    CLD_MASK_BAND,
    S1_BAND,
    S2_BAND,
)
from openeo_mmdc.dataset.dataclass import MMDCDF, ItemTensorMMDC
from openeo_mmdc.dataset.to_tensor import light_from_dataset2tensor
from openeo_mmdc.dataset.utils import (
    load_item_dataset_modality,
    merge_agera5_datasets,
)

my_logger = logging.getLogger(__name__)


def yearly_convert_to_tensor(
        c_mmdc_df: MMDCDF,
        item,
        list_years: list["str"],
        s2_drop_variable: list["str"] | None = None,
        s2_load_bands: list[str] | None = None,
        opt: Literal["all", "s2", "s1", "sentinel"] = "all",
        s2_band_mask: list | None = None,
        s2_max_ccp: float | None = None) -> list[ItemTensorMMDC]:
    """

    Args:
        seed ():
        s2_max_ccp ():
        all_transform ():
        crop_size ():
        s2_load_bands ():
        transform (): Transform applied to the torch tensor, for instance rescaling
        max_len ():
        crop_type ():
        s2_drop_variable ():
        c_mmdc_df ():
        item ():
        opt (): indicates which modality to load

    Returns:

    """

    if s2_load_bands is None:
        s2_load_bands = S2_BAND
    if s2_band_mask is None:
        s2_band_mask = CLD_MASK_BAND

    assert opt == "s2s1"
    s2_dataset = load_item_dataset_modality(
        mod_df=c_mmdc_df.s2,
        item=item,
        drop_variable=s2_drop_variable,
        load_variables=s2_load_bands + s2_band_mask,
        s2_max_ccp=s2_max_ccp,
    )

    s1_asc_dataset = load_item_dataset_modality(
        mod_df=c_mmdc_df.s1_asc, item=item
    )  # TODO maybe use drop_var depends if we want to keep the angle ...

    l_out = []
    s2_datasets = slice_year_dataset(s2_dataset, list_years)
    s1_datasets = slice_year_dataset(s1_asc_dataset, list_years)
    for j, year in enumerate(list_years):

        item = ItemTensorMMDC(s2=light_from_dataset2tensor(
            s2_datasets[j],
            band_cld=s2_band_mask,
            load_variable=s2_load_bands,
        ),
                              s1_asc=light_from_dataset2tensor(
                                  s1_datasets[j],
                                  load_variable=S1_BAND,
                                  dtype=torch.float32))
        l_out += [item]

    return l_out


def convert_to_tensor(
    c_mmdc_df: MMDCDF,
    item,
    s2_drop_variable: list["str"] | None = None,
    s2_load_bands: list[str] | None = None,
    opt: Literal["all", "s2", "s1", "sentinel"] = "all",
    s2_band_mask: list | None = None,
    s2_max_ccp: float | None = None,
) -> ItemTensorMMDC:
    """

    Args:
        seed ():
        s2_max_ccp ():
        all_transform ():
        crop_size ():
        s2_load_bands ():
        transform (): Transform applied to the torch tensor, for instance rescaling
        max_len ():
        crop_type ():
        s2_drop_variable ():
        c_mmdc_df ():
        item ():
        opt (): indicates which modality to load

    Returns:

    """
    out = {}
    if s2_load_bands is None:
        s2_load_bands = S2_BAND
    if s2_band_mask is None:
        s2_band_mask = CLD_MASK_BAND
    if opt in ("all", "s2", "s1s2"):
        s2_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s2,
            item=item,
            drop_variable=s2_drop_variable,
            load_variables=s2_load_bands + s2_band_mask,
            s2_max_ccp=s2_max_ccp,
        )
        out["s2"] = light_from_dataset2tensor(
            s2_dataset,
            band_cld=s2_band_mask,
            load_variable=s2_load_bands,
        )

    if opt in ("all", "s1"):
        s1_des_dataset = load_item_dataset_modality(mod_df=c_mmdc_df.s1_desc,
                                                    item=item)
        out["s1_desc"] = light_from_dataset2tensor(s1_des_dataset,
                                                   load_variable=S1_BAND,
                                                   dtype=torch.float32)
    if opt in ("all", "s1s2", "s1"):
        s1_asc_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s1_asc, item=item
        )  # TODO maybe use drop_var depends if we want to keep the angle ...
        out["s1_asc"] = light_from_dataset2tensor(s1_asc_dataset,
                                                  load_variable=S1_BAND,
                                                  dtype=torch.float32)

    if opt == "all":
        dem_dataset = load_item_dataset_modality(mod_df=c_mmdc_df.dem,
                                                 item=item,
                                                 s2_max_ccp=None,
                                                 dtype=torch.float32)
        out["dem"] = light_from_dataset2tensor(dem_dataset, )
        l_agera5_df = [
            c_mmdc_df.dew_temp,
            c_mmdc_df.temp_max,
            c_mmdc_df.temp_mean,
            c_mmdc_df.prec,
            c_mmdc_df.sol_rad,
            c_mmdc_df.temp_min,
            c_mmdc_df.sol_rad,
            c_mmdc_df.val_press,
        ]
        agera5dataset = merge_agera5_datasets(l_agera5_df, item)
        out["agera5"] = light_from_dataset2tensor(agera5dataset,
                                                  dtype=torch.float32)
    return ItemTensorMMDC(**out)


def slice_year_dataset(ds: xarray.Dataset,
                       list_years: list[str]) -> list[xarray.Dataset]:
    l_ds = []

    for year in list_years:

        sub_ds = ds.sel(t=slice(f"{year}-01-01", f"{year}-12-31"))
        assert len(sub_ds.t) > 0, "No sliced found at {year}"
        my_logger.debug(f"DOY year {sub_ds.t} at year {year}")
        l_ds += [sub_ds]
    return l_ds
