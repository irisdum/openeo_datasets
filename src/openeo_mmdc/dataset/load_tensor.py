import logging
import random
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch import Tensor, nn

from openeo_mmdc.dataset.dataclass import (
    MMDC_MAXLEN,
    PT_MMDC_DF,
    ItemTensorMMDC,
    MaskMod,
    ModTransform,
    OneMod,
)
from openeo_mmdc.dataset.to_tensor import crop_tensor, get_crop_idx

my_logger = logging.getLogger(__name__)


def create_mmcd_tensor_df(
    path_directory: str,
    s2_tile,
    modalities: list[Literal["s2", "s1_asc", "s1_desc", "dem", "agera5"]],
    format=".pt",
) -> PT_MMDC_DF:
    out_mod = {}
    for mod in modalities:
        out_mod[mod] = create_mod_df_tensor(
            path_dir=path_directory,
            s2_tile=s2_tile,
            modality=mod,
            format=format,
        )
    return PT_MMDC_DF(**out_mod)


def create_mod_df_tensor(
    path_dir: str, s2_tile, modality: str, format=".pt"
) -> pd.DataFrame:
    assert Path(path_dir).exists(), f"{path_dir} not found"
    print(modality)
    l_df = []
    for tile in s2_tile:
        pattern = f"{tile}/*{modality}{format}"
        l_s2 = [
            p for p in Path(path_dir).rglob(pattern)
        ]  # extract all s2 tiles
        l_df += [
            pd.DataFrame(
                create_dict_one_sits(
                    path_sits=path, mod=modality, s2_tile=tile
                )
            )
            for path in l_s2
        ]
        assert l_s2, f"No image found at {pattern} at {path_dir}"
    my_logger.debug(l_df)
    final_df = pd.concat(l_df, ignore_index=True)
    final_df = final_df.sort_values(["patch_id", "s2_tile"], ascending=True)
    return final_df


def create_dict_one_sits(path_sits: Path, mod: str, s2_tile: str):
    patch_id = path_sits.name.split("openEO_")[-1][:2]
    return [
        {
            "mod": mod,
            "patch_id": patch_id,
            "s2_tile": s2_tile,
            "sits_path": path_sits,
        }
    ]


def crop_spat_temp(
    tensor: Tensor,
    x: int,
    y: int,
    crop_size: int,
    list_t,
    padding_val: int = 0,
):
    assert (
        len(tensor.shape) == 4
    ), f"expected tensor type c,t,h,w got {tensor.shape}"
    cropped_tensor = crop_tensor(tensor, x, y, crop_size)

    if list_t is not None:
        cropped_tensor = cropped_tensor[:, list_t, ...]
    if padding_val:
        my_logger.debug(f"Apply padding of dim {padding_val}")
        padd_shape = list(cropped_tensor.shape)
        padd_shape[1] = padding_val
        padd_zeros = torch.zeros(tuple(padd_shape))
        cropped_tensor = torch.cat([cropped_tensor, padd_zeros], 1)
    return cropped_tensor


def load_mmdc_sits(
    c_mmdc_df: PT_MMDC_DF,
    item,
    crop_size: int,
    crop_type: Literal["Center", "Random"],
    max_len: MMDC_MAXLEN,  # TOD0 maybe add a specific len for each modality
    opt: Literal["all", "s2", "s1", "sentinel"] = "all",
    all_transform: None | ModTransform = None,
    seed: int | None = None,
) -> ItemTensorMMDC:
    out = {}
    if all_transform is None:
        all_transform = ModTransform()
    if opt in ("all", "s2"):
        temp_seed = seed + 0 if (seed is not None) else None

        out["s2"] = load_sits(
            c_mmdc_df.s2,
            item=item,
            transform=all_transform.s2.transform,
            crop_size=crop_size,
            crop_type=crop_type,
            max_len=max_len.s2,
            seed=temp_seed,
        )
    if opt in ("s1", "all"):
        temp_seed = seed + 1 if (seed is not None) else None
        out["s1_asc"] = load_sits(
            c_mmdc_df.s1_asc,
            item=item,
            transform=all_transform.s1_asc.transform,
            crop_size=crop_size,
            crop_type=crop_type,
            max_len=max_len.s1_asc,
            seed=temp_seed,
        )
        out["s1_desc"] = load_sits(
            c_mmdc_df.s1_desc,
            item=item,
            transform=all_transform.s1_desc.transform,
            crop_size=crop_size,
            crop_type=crop_type,
            max_len=max_len.s1_desc,
            seed=temp_seed,
        )
    if opt == "all":
        temp_seed_dem = seed + 3 if (seed is not None) else None
        out["dem"] = load_sits(
            c_mmdc_df.dem,
            item=item,
            transform=all_transform.dem.transform,
            crop_size=crop_size,
            crop_type=crop_type,
            max_len=None,
            seed=temp_seed_dem,
        )
        temp_seed = seed + 4 if (seed is not None) else None
        out["agera5"] = load_sits(
            c_mmdc_df.agera5,
            item=item,
            transform=all_transform.agera5.transform,
            crop_size=crop_size,
            crop_type=crop_type,
            max_len=max_len.agera5,
            seed=temp_seed,
        )
    return ItemTensorMMDC(**out)


def load_sits(
    df_mod: pd.DataFrame,
    item: int,
    transform: nn.Module,
    crop_size: int,
    crop_type: Literal["Center", "Random"],
    max_len: None | int,
    seed: int | None = None,
) -> OneMod:
    sits_series = df_mod.iloc[item]
    sits_obj: OneMod = torch.load(sits_series["sits_path"])
    shape_sits = sits_obj.sits.shape
    if transform:  # TODO move that onto the GPU ?
        sits_obj.sits = transform(sits_obj.sits)
    row, col = shape_sits[-1], shape_sits[-2]
    x, y = get_crop_idx(
        rows=row, cols=col, crop_size=crop_size, crop_type=crop_type
    )
    if seed is not None:
        random.seed(seed)
    if max_len is not None:
        temp_idx = sorted(
            random.sample([i for i in range(shape_sits[1])], max_len)
        )
        if max_len > shape_sits[1]:
            my_logger.debug(
                f"sits temporal dim {shape_sits[1]} is lower than max len"
                f" {max_len}"
            )
            temp_idx = [i for i in range(max_len)]

    else:
        temp_idx = None

    crop_sits = crop_spat_temp(sits_obj.sits, x, y, crop_size, temp_idx)
    my_logger.debug(sits_obj.mask.mask_cld.shape)
    my_logger.debug(sits_obj.mask.mask_nan[None, ...].shape)

    crop_mask_cld = crop_spat_temp(
        sits_obj.mask.mask_cld, x, y, crop_size, temp_idx
    )
    crop_mask_scl = crop_spat_temp(
        sits_obj.mask.mask_slc, x, y, crop_size, temp_idx
    )

    crop_nan_mask = crop_spat_temp(
        sits_obj.mask.mask_nan[None, ...], x, y, crop_size, temp_idx
    )
    # padd_mask=crop_spat_temp(torch.ones(shape_sits),x,y,crop_size,temp_idx)
    cropped_mask = MaskMod(
        mask_cld=crop_mask_cld,
        mask_slc=crop_mask_scl,
        mask_nan=crop_nan_mask,  # ,padd_mask=padd_mask.bool()
    )
    temp_cropped_doy = sits_obj.doy[temp_idx]
    # if padding_val>0:
    #     temp_cropped_doy=F.pad(temp_cropped_doy,(padding_val,1))
    return OneMod(sits=crop_sits, doy=temp_cropped_doy, mask=cropped_mask)
