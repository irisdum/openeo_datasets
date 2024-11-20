import logging
import random
from pathlib import Path

import pandas as pd
import xarray
from distributed import Client

from openeo_mmdc.constant.torch_dataloader import S2_BAND
from openeo_mmdc.dataset.utils import order_dataset_vars

my_logger = logging.getLogger(__name__)


def compute_stats_one_mod(
    path_dir,
    mod,
    band_list: list[str],
    ex_dir: str,
    max_img=2,
    qmin: float = 0.05,
    qmax: float = 0.95,
):
    list_all_files = [f for f in Path(path_dir).rglob(f"*/**/{mod}*/*clip.nc")]
    idx_file = random.sample([i for i in range(len(list_all_files))], max_img)
    l_open_file = [list_all_files[idx] for idx in idx_file]
    my_logger.info(l_open_file)
    global_dataset = xarray.open_mfdataset(
        l_open_file,
        mask_and_scale=False,
        chunks={"t": "500MB", "x": "500MB", "y": "500MB"},
        cache=False,
    )
    my_logger.info(global_dataset)

    global_dataset = order_dataset_vars(
        global_dataset, list_vars_order=band_list
    )
    # print(global_dataset)
    array = global_dataset.to_array()
    # array=array.stack(z=('t','x','y'))
    med = array.median(dim=["t", "x", "y"])
    s_med = pd.Series(dict(zip(band_list, med.values)))
    s_med.name = "med"
    # print(s_med)
    qmin = array.quantile(q=qmin, dim=["t", "x", "y"])
    s_qmin = pd.Series(dict(zip(band_list, qmin.values)))
    s_qmin.name = "qmin"
    qmax = array.quantile(q=qmax, dim=["t", "x", "y"])
    s_qmax = pd.Series(dict(zip(band_list, qmax.values)))
    s_qmax.name = "qmax"
    # print(med.values)
    print(f"qmin {qmin.values}\n med {med.values}\n qmax {qmax.values}")
    # stats_serie=[]
    df_stats = pd.concat([s_med, s_qmin, s_qmax], axis=1).T
    # print(df_stats)
    df_stats.to_csv(Path(ex_dir).joinpath(f"dataset_{mod}.csv"))


if __name__ == "__main__":
    PATH_DIR = "/home/ad/dumeuri/DeepChange/MMDC_OE/train/"

    client = Client(n_workers=1)
    compute_stats_one_mod(
        path_dir=PATH_DIR,
        ex_dir=PATH_DIR,
        mod="Sentinel2",
        band_list=S2_BAND,
        max_img=10,
    )
