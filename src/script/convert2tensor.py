import logging
from pathlib import Path

import hydra
import torch

from openeo_mmdc.dataset.convert import convert_to_tensor
from openeo_mmdc.dataset.utils import build_dataset_info

my_logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config/", config_name="convert.yaml")
def main(config):
    directory = config.directory
    Path(config.ex_dir).mkdir(exist_ok=True)
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
    for item in range(len(c_mmdc_df.s2)):
        item_series = c_mmdc_df.s2.iloc[item]
        tile = item_series["s2_tile"]
        patch_id = item_series["patch_id"]
        ex_path = Path(config.ex_dir).joinpath(
            f"{tile}/Patch_item_{item}_id_{patch_id}.pt"
        )
        if not ex_path.exists():
            Path(config.ex_dir).joinpath(tile).mkdir(exist_ok=True)
            out_transform = convert_to_tensor(
                c_mmdc_df, item, s2_max_ccp=config.s2_max_ccp, opt="all"
            )
            torch.save(out_transform, ex_path)
            torch.save(
                c_mmdc_df,
                Path(config.ex_dir).joinpath("tiles_descriptions.pt"),
            )
        else:
            my_logger.info(f"We have already created tensor {ex_path}")


if __name__ == "__main__":
    main()
