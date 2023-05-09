import json
import time
from dataclasses import dataclass
from pathlib import Path

import openeo
from openeo import BatchJob, DataCube

from openeo_mmdc.build_datacube import (
    OutRunJob,
    download_agora_per_band,
    download_dem,
    download_s1,
    download_s2,
)
from openeo_mmdc.constant.dataset import FEATURES_VAL


@dataclass
class OutputS2:
    collection: DataCube
    job_dir: str
    job: BatchJob


def pull_and_download(jobs: list[OutRunJob], download: bool = False):
    jobs = jobs.copy()
    while jobs:
        for job in jobs:
            job_id = job.job
            out_dir = job.outdir
            if job_id.status() == "finished":
                print(f"Job {job_id} is ready. Downloading to {out_dir}")
                if download:
                    job_id.get_results().download_files(out_dir)
                    jobs.remove(job)
            else:
                print(f"Job {job_id} is not ready.")
        time.sleep(30)


if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    print(c.describe_account())
    year = "2020"
    TIMERANGE = [f"{year}-01-01", f"{year}-12-31"]
    assert len(FEATURES_VAL) > 0, "No geoson file found"
    features = FEATURES_VAL[1]
    tile = features.name.split("_")[-1].split(".")[0]
    features = str(features)
    print(Path.cwd())
    if features.endswith("json") and features.startswith("/"):
        with open(features) as feat:
            print(feat)
            features = json.load(feat)
    print(features)
    output_s2 = download_s2(
        c,
        features=features,
        tile=tile,
        temporal_extent=TIMERANGE,
        year=year,
        run=False,
    )
    job_s1_asc = download_s1(
        c,
        collection_s2=output_s2.collection,
        orbit="ASCENDING",
        features=features,
        tile=tile,
        temporal_extent=TIMERANGE,
        year=year,
    )
    job_s1_des = download_s1(
        c,
        collection_s2=output_s2.collection,
        orbit="DESCENDING",
        features=features,
        tile=tile,
        temporal_extent=TIMERANGE,
        year=year,
    )
    agera5_bands = [
        "dewpoint-temperature",
        "precipitation-flux",
        "solar-radiation-flux",
        "temperature-max",
        "temperature-mean",
        "temperature-min",
        "vapour-pressure",
        "wind-speed",
    ]
    for b in agera5_bands:
        job_agora = download_agora_per_band(
            c,
            collection_s2=output_s2.collection,
            features=features,
            tile=tile,
            temporal_extent=TIMERANGE,
            year=year,
            bands=[b],
        )
    job_dem = download_dem(
        c,
        collection_s2=output_s2.collection,
        features=features,
        tile=tile,
        year=year,
    )
    pull_and_download(
        [job_agora],
        download=False,
    )  # job_s1_asc, job_s1_des, job_agora, job_dem #job_s1_asc, job_s1_des, job_dem
    # poll_and_download([job_s2, job_s1, job_dem, job_agera5])
