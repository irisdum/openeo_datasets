import logging
import os
from pathlib import Path

import dask
import hydra
import torch
import xarray as xr
from dask.distributed import Client

from openeo_mmdc.dataset.convert import convert_to_tensor
from openeo_mmdc.dataset.utils import build_dataset_info

my_logger = logging.getLogger(__name__)


def save_per_mod(mods: list, mmdc_sits, ex_dir: str | Path, suffix: str):
    for mod in mods:
        if mod == "s2":
            my_logger.debug(Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
            torch.save(mmdc_sits.s2,
                       Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
        elif "s1_asc" == mod:
            torch.save(mmdc_sits.s1_asc,
                       Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
        elif "s1_desc" == mod:
            torch.save(mmdc_sits.s1_desc,
                       Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
        elif "dem" == mod:
            torch.save(mmdc_sits.dem,
                       Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
        elif "agera5" == mod:
            torch.save(mmdc_sits.agera5,
                       Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
        else:
            raise NotImplementedError(mod)


def convert(c_mmdc_df, config, item, mod_df):
    item_series = c_mmdc_df.s2.iloc[item]
    tile = item_series["s2_tile"]
    patch_id = item_series["patch_id"][:-3]
    ex_path = Path(config.ex_dir).joinpath(
        f"{tile}/Patch_item_id_{patch_id}_{mod_df[0]}.pt")
    if not ex_path.exists():
        try:
            Path(config.ex_dir).joinpath(tile).mkdir(exist_ok=True)
            out_transform = convert_to_tensor(c_mmdc_df,
                                              item,
                                              s2_max_ccp=config.s2_max_ccp,
                                              opt=config.opt)
            if config.opt == "all":
                mod = ["s2", "s1_asc", "s1_desc", "dem", "agera5"]
            elif config.opt == "s1":
                mod = ["s1_asc", "s1_desc"]
            elif config.opt == "s2":
                mod = ["s2"]
            elif config.opt == "sentinel":
                mod = ["s2", "s1_asc", "s1_desc"]
            save_per_mod(
                mods=mod,
                mmdc_sits=out_transform,
                ex_dir=Path(config.ex_dir).joinpath(tile),
                suffix=f"Patch_id_{patch_id}",
            )
            # torch.save(out_transform, ex_path)

            my_logger.info(f"Create {ex_path}")
        except:
            pass

    else:
        my_logger.info(f"We have already created tensor {ex_path}")
    return item


@hydra.main(config_path="../../config/", config_name="convert.yaml")
def main(config):
    directory = config.directory
    Path(config.ex_dir).mkdir(exist_ok=True, parents=True)
    if config.mod_df is None:
        mod_df = [
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
        ]
    else:
        mod_df = config.mod_df
    c_mmdc_df = build_dataset_info(path_dir=directory,
                                   l_tile_s2=config.s2_tile,
                                   list_modalities=mod_df)
    torch.save(
        c_mmdc_df,
        Path(config.ex_dir).joinpath("tiles_descriptions.pt"),
    )
    # l_out = [
    #     convert(c_mmdc_df, config, item, mod_df)
    #     for item in range(len(c_mmdc_df.s2))
    # ]
    res_item = []
    for item in range(len(c_mmdc_df.s2)):
        #item_out = dask.delayed(convert)(c_mmdc_df, config, item, mod_df)
        item_out = convert(c_mmdc_df, config, item, mod_df)
        res_item.append(item_out)
    #results = dask.compute(*res_item)
    return res_item


if __name__ == "__main__":

    main()
