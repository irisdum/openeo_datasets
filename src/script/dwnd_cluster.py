"""
Dwnd tiles in cluster
"""
import os
from pathlib import Path

import openeo
import wget

from openeo_mmdc.build_datacube import sub_dir_name


def extract_time(dict_metadata) -> str:
    time_range = dict_metadata["properties"]["card4l:processing_chain"][
        "process_graph"
    ]["loadcollection1"]["arguments"]["temporal_extent"]
    return time_range[0].split("-")[0]


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
    wget.download(url=link, out=ex_dir)


def extract_suffix(dict_metadata: dict):
    print(dict_metadata["properties"].keys())
    return dict_metadata["properties"]["title"]


def main(list_id: list[str], c, ex_dir):
    """

    Args:
        list_id (): is a the list of doi we want to use

    Returns:

    """
    for job_id in list_id:
        res = c.job(job_id).get_results()
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
        for roi in assets_metadata.keys():
            print(roi)
            if not Path(os.path.join(out_dir, roi)).exists():
                dwnd_file(assets_metadata[roi]["href"], ex_dir=out_dir)


if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    list_jobs = ["vito-j-e9178da12c0b4c9f8a586e071e871aa2"]
    main(list_jobs, c, ex_dir="/home/dumeuri/Documents/dataset/MMDC_OE")
