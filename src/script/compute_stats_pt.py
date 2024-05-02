import logging
import random
from pathlib import Path

import dask
import hydra
import pandas as pd
import torch
from einops import rearrange

from openeo_mmdc.constant.torch_dataloader import S2_BAND

my_logger = logging.getLogger(__name__)


def get_tensor_pix_bands(tensor_path, crop_size):
    my_logger.debug(tensor_path)
    tensor = torch.load(tensor_path)
    my_logger.debug(type(tensor))
    sub_tensor = rearrange(
        tensor.sits[..., :crop_size, :crop_size], "c t h w ->  (t h w) c "
    )
    return sub_tensor


@hydra.main(config_path="../../config/", config_name="stats.yaml")
def main(config):
    l_sel_img = []
    for s2_tile in config.s2_tiles:
        tmpt_dir = Path(config.path_dir, s2_tile)
        l_img = [p for p in tmpt_dir.rglob(f"*{config.mod}.pt")]
        my_logger.debug(l_img)
        assert l_img, f"No images found at {tmpt_dir}"
        l_sel_img += random.sample(l_img, config.max_img)
    my_logger.debug(l_sel_img)
    l_tensor = [
        dask.delayed(get_tensor_pix_bands)(path, config.crop_size)
        for path in l_sel_img
    ]
    with dask.config.set(scheduler="processes"):
        l_computed = dask.compute(*l_tensor)
    my_logger.debug(f"{[elem.shape for elem in l_computed]}")
    all_tensors = torch.cat(l_computed, dim=0)
    med = torch.median(all_tensors, dim=0).values
    my_logger.debug(f"{med}")
    s_med = pd.Series(dict(zip(S2_BAND, [float(med_val) for med_val in med])))
    s_med.name = "med"
    qmin = torch.quantile(all_tensors, q=config.qmin, dim=0)
    my_logger.debug(f"{qmin}")
    s_qmin = pd.Series(dict(zip(S2_BAND, [float(q_val) for q_val in qmin])))
    s_qmin.name = "qmin"
    qmax = torch.quantile(all_tensors, q=config.qmax, dim=0)
    s_qmax = pd.Series(dict(zip(S2_BAND, [float(q_val) for q_val in qmax])))
    s_qmax.name = "qmax"
    df_stats = pd.concat([s_med, s_qmin, s_qmax], axis=1).T
    # print(df_stats)
    path_out = Path(config.ex_dir).joinpath(f"dataset_{config.mod}.csv")
    df_stats.to_csv(path_out)
    my_logger.info(f"Save csv at {path_out}")


if __name__ == "__main__":
    main()
