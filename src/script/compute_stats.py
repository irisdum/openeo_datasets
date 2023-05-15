import random
from pathlib import Path

import xarray

from openeo_mmdc.dataset.utils import order_dataset_vars


def compute_stats(path_dir, mod, max_img=2, band_list: list[str] = None):
    list_all_files = [f for f in Path(path_dir).rglob(f"*/**/{mod}*/*clip.nc")]
    idx_file = random.sample([i for i in range(len(list_all_files))], max_img)
    l_open_file = [list_all_files[idx] for idx in idx_file]
    print(l_open_file)
    global_dataset = xarray.open_mfdataset(l_open_file, combine="nested")
    if band_list is not None:
        global_dataset = order_dataset_vars(
            global_dataset, list_vars_order=None
        )
    print(global_dataset)


if __name__ == "__main__":
    PATH_DIR = "/media/dumeuri/DATA/Data/Iris/MMDC_OE"
    compute_stats(path_dir=PATH_DIR, mod="Sentinel2")
