import logging
from typing import Literal

from openeo_mmdc.constant.torch_dataloader import CLD_MASK_BAND, S2_BAND
from openeo_mmdc.dataset.dataclass import MMDCDF, ItemTensorMMDC, ModTransform
from openeo_mmdc.dataset.to_tensor import from_dataset2tensor
from openeo_mmdc.dataset.utils import (
    load_item_dataset_modality,
    merge_agera5_datasets,
)

my_logger = logging.getLogger(__name__)


def mmdc_sits(
    c_mmdc_df: MMDCDF,
    item,
    s2_drop_variable: list["str"] | None,
    s2_load_bands: list[str] | None,
    crop_size: int,
    crop_type: Literal["Center", "Random"],
    max_len: int,
    opt: Literal["all", "s2", "s1", "sentinel"] = "all",
    all_transform: None | ModTransform = None,
    s2_band_mask: list | None = None,
    s2_max_ccp: float | None = None,
    seed: int | None = None,
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
    my_logger.debug(f"opt {opt}")
    if opt in ("all", "s2"):
        s2_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s2,
            item=item,
            drop_variable=s2_drop_variable,
            load_variables=s2_load_bands + s2_band_mask,
            s2_max_ccp=s2_max_ccp,
        )
        my_logger.debug(f"band mask {s2_band_mask} bands {s2_load_bands}")
        out["s2"] = from_dataset2tensor(
            s2_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.s2.transform,
            band_cld=list(s2_band_mask),
            load_variable=s2_load_bands,
            seed=seed,
        )

    if opt in ("all", "s1"):
        s1_asc_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s1_asc, item=item
        )  # TODO maybe use drop_var depends if we want to keep the angle ...
        out["s1_asc"] = from_dataset2tensor(
            s1_asc_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.s1_asc.transform,
            seed=seed,
        )
        s1_des_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s1_desc, item=item
        )
        out["s1_desc"] = from_dataset2tensor(
            s1_des_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.s1_desc.transform,
            seed=seed,
        )
    if opt == "all":
        dem_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.dem, item=item
        )
        out["dem"] = from_dataset2tensor(
            dem_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.dem.transform,
            seed=seed,
        )
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
        out["agera5"] = from_dataset2tensor(
            merge_agera5_datasets(l_agera5_df, item),
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.agera5.transform,
            seed=seed,
        )
    return ItemTensorMMDC(**out)
