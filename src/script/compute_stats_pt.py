import random
from pathlib import Path

import dask
import hydra
import pandas as pd
import torch
from einops import rearrange

from openeo_mmdc.constant.torch_dataloader import S2_BAND


def get_tensor_pix_bands(tensor_path, crop_size):
    tensor = torch.load(tensor_path).sits
    sub_tensor = rearrange(
        tensor[..., :crop_size, :crop_size], "t c h w ->  (t h w) c "
    )
    return sub_tensor


@hydra.main(config_path="../../config/", config_name="stats.yaml")
def main(config):
    l_sel_img = []
    for s2_tile in config.s2_tiles:
        tmpt_dir = Path(config.path_dir, s2_tile)
        l_img = [p for p in tmpt_dir.rglob(f"*/**/*{config.mod}.pt")]
        assert l_img, f"No images found at {tmpt_dir}"
        l_sel_img += [
            random.sample([i for i in range(len(l_img))], config.max_img)
        ]
    l_tensor = dask.compute(
        [
            dask.delayed(get_tensor_pix_bands)(path, config.crop_size)
            for path in l_sel_img
        ]
    )
    all_tensors = torch.cat(l_tensor, dim=0)
    med = torch.median(all_tensors, dim=0)
    s_med = pd.Series(dict(zip(S2_BAND, [float(med_val) for med_val in med])))
    s_med.name = "med"
    qmin = torch.quantile(all_tensors, q=config.qmin, dim=0)
    s_qmin = pd.Series(dict(zip(S2_BAND, [float(q_val) for q_val in qmin])))
    s_qmin.name = "qmin"
    qmax = torch.quantile(all_tensors, q=config.qmax, dim=0)
    s_qmax = pd.Series(dict(zip(S2_BAND, [float(q_val) for q_val in qmax])))
    s_qmax.name = "qmax"
    df_stats = pd.concat([s_med, s_qmin, s_qmax], axis=1).T
    # print(df_stats)
    df_stats.to_csv(Path(config.ex_dir).joinpath(f"dataset_{config.mod}.csv"))


if __name__ == "__main__":
    main()
