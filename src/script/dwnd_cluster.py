"""
Dwnd tiles in cluster
"""
import os
from pathlib import Path
import getpass
import dask
import openeo
from dask.distributed import Client
import requests
from openeo_mmdc.build_datacube import sub_dir_name
from openeo_mmdc.open import open_job_df


def extract_time(dict_metadata) -> str | None:
    time_range = dict_metadata["properties"]["card4l:processing_chain"][
        "process_graph"]["loadcollection1"]["arguments"]["temporal_extent"]
    if time_range is not None:
        return time_range[0].split("-")[0]
    return None


def extracts2_tile(dict_metadata: dict):
    print(
        dict_metadata["properties"]["card4l:processing_chain"]["process_graph"]
        ["filterspatial1"]["arguments"]["geometries"]["features"][0].keys())
    return dict_metadata["properties"]["card4l:processing_chain"][
        "process_graph"]["filterspatial1"]["arguments"]["geometries"][
            "features"][0]["properties"]["Name"]


def dwnd_file(link, ex_dir,roi):
    print(link)
    cmd = f"curl -k -O --output-dir {ex_dir} {link} "
    # os.system("curl -V")
    # p = subprocess.Popen(cmd,shell=True)
    r = requests.get(link, allow_redirects=True)
    
    open(os.path.join(ex_dir,roi), 'wb').write(r.content)
    print(f"save {roi}")
#    os.system(cmd)
    # wget.download(url=link, out=ex_dir
    return link


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
                            dwnd_file(assets_metadata[roi]["href"],
                                      ex_dir=out_dir,roi=roi))
                    ]
                else:
                    print(f"file {roi} exists")
        except openeo.rest.OpenEoApiError:
            print(f"OPenAI rest error {res} ")


if __name__ == "__main__":
    user = getpass.getuser()
    pw = getpass.getpass()
    os.environ['http_proxy'] = "http://{}:{}@proxy-surf.loc.cnes.fr:8050".format(user,pw)
    os.environ['https_proxy'] = "http://{}:{}@proxy-surf.loc.cnes.fr:8050".format(user,pw)

    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    client = Client(threads_per_worker=4, n_workers=1)
    # list_jobs = ["vito-j-e9178da12c0b4c9f8a586e071e871aa2"]
    list_jobs = open_job_df(
        "/home/ad/dumeuri/scratch/datasets/MMDC_OE/val/reporting_export_20230509.csv"
    )
    main(list_jobs, c, ex_dir="/home/ad/dumeuri/scratch/datasets/MMDC_OE/val")
