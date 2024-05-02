import logging
from pathlib import Path

import hydra
import torch

from openeo_mmdc.dataset.convert import (
    convert_to_tensor,
    yearly_convert_to_tensor,
)
from openeo_mmdc.dataset.dataclass import MaskMod, OneMod
from openeo_mmdc.dataset.utils import build_dataset_info

my_logger = logging.getLogger(__name__)


def save_per_mod(
    mods: list, mmdc_sits, ex_dir: str | Path, suffix: str, crop_size=128
):
    for mod in mods:
        if mod == "s2":
            print(Path(ex_dir).joinpath(f"{suffix}_{mod}.pt"))
            l_tensor = crop_mod(mmdc_sits.s2, crop_size=crop_size)

            for i, tensor in enumerate(l_tensor):
                print(f"image id{i}")
                torch.save(
                    tensor,
                    Path(ex_dir).joinpath(f"{suffix}_id{i}_{mod}.pt"),
                )
        elif "s1_asc" == mod:
            l_tensor = crop_mod(mmdc_sits.s1_asc, crop_size=crop_size)
            for i, tensor in enumerate(l_tensor):
                torch.save(
                    tensor,
                    Path(ex_dir).joinpath(f"{suffix}_id{i}_{mod}.pt"),
                )

        elif "s1_desc" == mod:
            l_tensor = crop_mod(mmdc_sits.s1_desc, crop_size=crop_size)
            for i, tensor in enumerate(l_tensor):
                torch.save(
                    tensor,
                    Path(ex_dir).joinpath(f"{suffix}_id{i}_{mod}.pt"),
                )

        elif "dem" == mod:
            l_tensor = crop_mod(mmdc_sits.dem, crop_size=crop_size)

            for i, tensor in enumerate(l_tensor):
                torch.save(
                    tensor,
                    Path(ex_dir).joinpath(f"{suffix}_id{i}_{mod}.pt"),
                )

        elif "agera5" == mod:
            l_tensor = crop_mod(mmdc_sits.agera5, crop_size=crop_size)

            for i, tensor in enumerate(l_tensor):
                torch.save(
                    tensor,
                    Path(ex_dir).joinpath(f"{suffix}_id{i}_{mod}.pt"),
                )

        else:
            raise NotImplementedError(mod)


def crop_mod(mod: OneMod, crop_size) -> list[OneMod]:
    l_sits = crop_tensor(mod.sits, crop_size)
    n_crops = len(l_sits)
    if mod.mask.mask_cld is not None:
        l_cld = crop_tensor(mod.mask.mask_cld, crop_size)
    else:
        l_cld = [None] * n_crops
    if mod.mask.mask_nan is not None:
        l_nan = crop_tensor(mod.mask.mask_nan, crop_size)
    else:
        l_nan = [None] * n_crops
    if mod.mask.mask_slc is not None:
        l_slc = crop_tensor(mod.mask.mask_slc, crop_size)
    else:
        l_slc = [None] * n_crops
    l_mod = []
    for i, tensor in enumerate(l_sits):
        maski = MaskMod(l_cld[i], l_nan[i], l_slc[i])
        mod = OneMod(tensor, mod.doy, mask=maski, true_doy=mod.true_doy)
        l_mod += [mod]
    return l_mod


def crop_tensor(tensor, crop_size):
    if tensor.shape[-1] == crop_size:
        return [tensor]
    assert tensor.shape[-1] % crop_size == 0, (
        f"impossible to crop image size {tensor.shape[-1]} should be devisible"
        f" by {crop_size}"
    )
    assert tensor.shape[-1] == tensor.shape[-2]
    n_crops = tensor.shape[-1] // crop_size
    l_tensor = []
    for n in range(n_crops):
        l_tensor += [
            tensor[
                ...,
                n * crop_size : (n + 1) * crop_size,
                n * crop_size : (n + 1) * crop_size,
            ]
        ]

    return l_tensor


def convert(c_mmdc_df, config, item, mod_df, crop_size=128):
    item_series = c_mmdc_df.s2.iloc[item]
    tile = item_series["s2_tile"]
    patch_id = item_series["patch_id"][:-3]
    pattern = f"{tile}/Patch_id_{patch_id}_*_{mod_df[0]}.pt"
    my_logger.debug(pattern)
    lfound = [i for i in Path(config.ex_dir).rglob(pattern)]
    if not lfound:
        #        with suppress(BaseException):
        Path(config.ex_dir).joinpath(tile).mkdir(exist_ok=True)
        out_transform = convert_to_tensor(
            c_mmdc_df, item, s2_max_ccp=config.s2_max_ccp, opt=config.opt
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
            suffix=f"Patch_id_{patch_id}",
            crop_size=crop_size,
        )
        # torch.save(out_transform, ex_path)

    else:
        ex_path = f"{tile}/Patch_item_id_{patch_id}_*_{mod_df[0]}.pt"
        my_logger.info(f"We have already created tensor {ex_path} {lfound}")
    return item


def yearly_convert(
    c_mmdc_df, config, item, mod_df, years: list[str], crop_size=128
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
        l_out_transform = yearly_convert_to_tensor(
            c_mmdc_df,
            item,
            list_years=years,
            s2_max_ccp=config.s2_max_ccp,
            opt="s2s1",
        )
        assert len(l_out_transform) == len(years)
        if config.opt == "all":
            mod = ["s2", "s1_asc", "s1_desc", "dem", "agera5"]
        elif config.opt == "s1":
            mod = ["s1_asc", "s1_desc"]
        elif config.opt == "s1s2":
            mod = ["s1_asc", "s2"]
        elif config.opt == "s2":
            mod = ["s2"]
        elif config.opt == "sentinel":
            mod = ["s2", "s1_asc", "s1_desc"]
        else:
            raise NotImplementedError
        for i, year in enumerate(years):
            save_per_mod(
                mods=mod,
                mmdc_sits=l_out_transform[i],
                ex_dir=Path(config.ex_dir).joinpath(tile),
                suffix=f"Patch_id_{patch_id}_{year}",
                crop_size=crop_size,
            )
            my_logger.debug(f"Saved Patch_id_{patch_id}_{year}")
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
    # l_out = [
    #     convert(c_mmdc_df, config, item, mod_df)
    #     for item in range(len(c_mmdc_df.s2))
    # ]
    res_item = []
    for item in range(len(c_mmdc_df.s2)):
        if config.years is not None:
            item_out = yearly_convert(
                c_mmdc_df,
                config,
                item,
                mod_df,
                years=config.years,
                crop_size=config.crop_size,
            )

        else:
            item_out = convert(
                c_mmdc_df, config, item, mod_df, crop_size=config.crop_size
            )
        res_item.append(item_out)
    # results = dask.compute(*res_item)
    return res_item


if __name__ == "__main__":
    main()
