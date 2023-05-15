import logging.config
from typing import Literal

from openeo_mmdc.constant.torch_dataloader import LOAD_VARIABLE
from openeo_mmdc.dataset.dataclass import MMDCDF, ItemTensorMMDC, ModTransform
from openeo_mmdc.dataset.to_tensor import from_dataset2tensor
from openeo_mmdc.dataset.utils import (
    load_item_dataset_modality,
    merge_agera5_datasets,
)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


def mmdc_sits(
    c_mmdc_df: MMDCDF,
    item,
    s2_drop_variable: list["str"],
    s2_load_variables: list[str] | None,
    crop_size: int,
    crop_type: Literal["Center", "Random"],
    max_len: int,
    opt: Literal["all", "s2", "s1", "sentinel"] = "all",
    all_transform: None | ModTransform = None,
) -> ItemTensorMMDC:
    """

    Args:
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
    if s2_load_variables is None:
        s2_load_variables = LOAD_VARIABLE
    if opt in ("all", "s2"):
        s2_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s2,
            item=item,
            drop_variable=s2_drop_variable,
            load_variables=s2_load_variables,
        )
        out["s2"] = from_dataset2tensor(
            s2_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.s2,
        )

        my_logger.debug(
            f"out s2 arra{from_dataset2tensor(s2_dataset, max_len).sits.shape}"
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
            transform=all_transform.s1_asc,
        )
        s1_des_dataset = load_item_dataset_modality(
            mod_df=c_mmdc_df.s1_desc, item=item
        )
        out["s1_desc"] = from_dataset2tensor(
            s1_des_dataset,
            max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.s1_desc,
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
            transform=all_transform.dem,
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
        out["agera5"] = merge_agera5_datasets(
            l_agera5_df,
            item,
            max_len=max_len,
            crop_size=crop_size,
            crop_type=crop_type,
            transform=all_transform.agera5,
        )
    return ItemTensorMMDC(**out)
