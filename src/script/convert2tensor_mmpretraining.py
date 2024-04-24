import logging
from pathlib import Path

import hydra
import torch

from openeo_mmdc.dataset.convert import convert_rd_time_crop_mm_sits
from openeo_mmdc.dataset.utils import build_dataset_info
from script.convert2tensor import save_per_mod

my_logger = logging.getLogger(__name__)


def convert_mm(
    c_mmdc_df, config, item, mod_df, crop_size=128, seed: int | None = None
):
    item_series = c_mmdc_df.s2.iloc[item]
    tile = item_series["s2_tile"]
    patch_id = item_series["patch_id"][:-3]
    pattern = f"{tile}/Patch_id_{patch_id}_*_{mod_df[0]}.pt"
    my_logger.debug(pattern)
    lfound = [i for i in Path(config.ex_dir).rglob(pattern)]
    if not lfound:
        #        with suppress(BaseException):
        Path(config.ex_dir).joinpath(tile).mkdir(exist_ok=True)
        out_transform = convert_rd_time_crop_mm_sits(
            c_mmdc_df,
            item,
            opt=config.opt,
            s2_max_ccp=config.s2_max_ccp,
            seed=seed,
            l_possible_months=config.l_possible_months,
        )
        if config.opt == "all":
            mod = ["s2", "s1_asc", "s1_desc", "dem", "agera5"]
        elif config.opt == "s1":
            mod = ["s1_asc", "s1_desc"]
        elif config.opt == "s2":
            mod = ["s2"]
        elif config.opt == "sentinel":
            mod = ["s2", "s1_asc", "s1_desc"]
        else:
            raise NotImplementedError
        save_per_mod(
            mods=mod,
            mmdc_sits=out_transform,
            ex_dir=Path(config.ex_dir).joinpath(tile),
            suffix=f"Patch_id_{patch_id}_seed{seed}",
            crop_size=crop_size,
        )
        # torch.save(out_transform, ex_path)

    else:
        ex_path = f"{tile}/Patch_item_id_{patch_id}_*_{mod_df[0]}.pt"
        my_logger.info(f"We have already created tensor {ex_path} {lfound}")
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
    c_mmdc_df = build_dataset_info(
        path_dir=directory, l_tile_s2=config.s2_tile, list_modalities=mod_df
    )
    torch.save(
        c_mmdc_df,
        Path(config.ex_dir).joinpath("tiles_descriptions.pt"),
    )
    res_item = []
    for item in range(len(c_mmdc_df.s2)):
        # item_out = dask.delayed(convert)(c_mmdc_df, config, item, mod_df)
        for rep in config.repeat:
            item_out = convert_mm(
                c_mmdc_df,
                config,
                item,
                mod_df,
                crop_size=config.crop_size,
                seed=item + rep,
            )
            res_item.append(item_out)
    # results = dask.compute(*res_item)
    return res_item


if __name__ == "__main__":
    main()
