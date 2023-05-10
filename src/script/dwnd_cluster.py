"""
Dwnd tiles in cluster
"""
import os
from pathlib import Path

import dask
import hydra
import openeo
from dask.distributed import Client
from omegaconf import DictConfig

from openeo_mmdc.build_datacube import sub_dir_name
from openeo_mmdc.open import open_job_df


def extract_time(dict_metadata) -> str | None:
    time_range = dict_metadata["properties"]["card4l:processing_chain"][
        "process_graph"
    ]["loadcollection1"]["arguments"]["temporal_extent"]
    if time_range is not None:
        return time_range[0].split("-")[0]
    return None


def extracts2_tile(dict_metadata: dict):
    print(
        dict_metadata["properties"]["card4l:processing_chain"][
            "process_graph"
        ]["filterspatial1"]["arguments"]["geometries"]["features"][0].keys()
    )
    return dict_metadata["properties"]["card4l:processing_chain"][
        "process_graph"
    ]["filterspatial1"]["arguments"]["geometries"]["features"][0][
        "properties"
    ][
        "Name"
    ]


def dwnd_file(link, ex_dir):
    print(link)
    cmd = f"curl -O --output-dir {ex_dir} {link} "
    # os.system("curl -V")
    # p = subprocess.Popen(cmd,shell=True)

    os.system(cmd)
    # wget.download(url=link, out=ex_dir
    return link


def extract_suffix(dict_metadata: dict):
    print(dict_metadata["properties"].keys())
    return dict_metadata["properties"]["title"]


@hydra.main(config_path="../../config/", config_name="dwnd.yaml")
def main(config: DictConfig, connection):
    """
    Args:
    Returns:
    """
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    Client(threads_per_worker=4, n_workers=1)
    ex_dir = config.ex_dir
    list_id = open_job_df(config.path_csv)
    for job_id in list_id:
        res = c.job(job_id).get_results()
        try:
            dict_metadata = res.get_metadata()
            print(dict_metadata.keys())
            year = extract_time(dict_metadata)
            tile = extracts2_tile(dict_metadata)
            suffix = extract_suffix(dict_metadata)
            assets_metadata = dict_metadata["assets"]
            sub_dir = sub_dir_name(suffix=suffix, tile=tile, year=year)
            print(sub_dir)
            # creating a new directory called pythondirectory
            out_dir = os.path.join(ex_dir, sub_dir)
            if not Path(out_dir).is_dir():
                print(out_dir)
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            l_out = []
            for roi in assets_metadata.keys():
                print(roi)
                if not Path(os.path.join(out_dir, roi)).exists():
                    l_out += [
                        dask.delayed(
                            dwnd_file(
                                assets_metadata[roi]["href"], ex_dir=out_dir
                            )
                        )
                    ]  # TODO not sure maybe change that ...
                else:
                    print(f"file {roi} exists")
        except openeo.rest.OpenEoApiError:
            print(f"OPenAI rest error {res} ")


if __name__ == "__main__":
    # list_jobs = ["vito-j-e9178da12c0b4c9f8a586e071e871aa2"]
    main()
