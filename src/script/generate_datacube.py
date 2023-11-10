import json
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import openeo
from omegaconf import DictConfig
from openeo import BatchJob, DataCube

from openeo_mmdc.build_datacube import (
    OutRunJob,
    download_agora_per_band,
    download_dem,
    download_s1,
    download_s2,
)


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


@hydra.main(config_path="../../config/", config_name="generate_datacube.yaml")
def main(config: DictConfig):
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    print(c.describe_account())
    year = str(config.year)
    if config.begin_date is None or config.end_date is None:
        TIMERANGE = [f"{config.year}-01-01", f"{config.year}-12-31"]
    else:
        TIMERANGE = [config.begin_date, config.end_date]
    print(TIMERANGE)
    FEAT = sorted(Path(config.geojson_dir).rglob(pattern=config.pattern))
    assert FEAT, "No geoson file found"
    features = FEAT[config.id]
    tile = features.name.split("_")[-1].split(".")[0]
    features = str(features)
    print(Path.cwd())
    if features.endswith("json") and features.startswith("/"):
        with open(features) as feat:
            print(feat)
            features = json.load(feat)
    # print(features)
    if config.opt in ("all", "s2"):
        run_s2 = True
    else:
        run_s2 = False
    output_s2 = download_s2(
        c,
        features=features,
        tile=tile,
        temporal_extent=TIMERANGE,
        year=year,
        run=run_s2,
        max_cc=config.max_cc,
    )
    if config.opt in ("all", "s1_asc"):
        download_s1(
            c,
            collection_s2=output_s2.collection,
            orbit="ASCENDING",
            features=features,
            tile=tile,
            temporal_extent=TIMERANGE,
            year=year,
        )
    if config.opt in ("all", "s1_desc"):
        download_s1(
            c,
            collection_s2=output_s2.collection,
            orbit="DESCENDING",
            features=features,
            tile=tile,
            temporal_extent=TIMERANGE,
            year=year,
        )
    if config.opt in ("all", "dem"):
        download_dem(
            c,
            collection_s2=output_s2.collection,
            features=features,
            tile=tile,
            year=year,
        )
    if config.opt in ("all", "agera5"):
        if config.agera_band is None:
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
        else:
            agera5_bands = config.agera_band
        for b in agera5_bands:
            download_agora_per_band(
                c,
                collection_s2=output_s2.collection,
                features=features,
                tile=tile,
                temporal_extent=TIMERANGE,
                year=year,
                bands=[b],
            )


if __name__ == "__main__":
    main()
