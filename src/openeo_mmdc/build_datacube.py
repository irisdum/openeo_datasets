import logging.config
from dataclasses import dataclass
from pathlib import Path

from openeo import BatchJob, DataCube

from openeo_mmdc.constant.dataset import (
    D_AGERA5_BAND_NAME,
    FEATURES_VAL,
    OUTDIR,
    S2_BANDS,
    S2_COLLECTION,
    TIMERANGE,
)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


@dataclass
class OutRunJob:
    job: BatchJob
    outdir: Path
    collection: DataCube


def run_job(
    datacube: DataCube,
    title: str,
    description: str,
    subdir: str,
    features,
    job_options=None,
    run=True,
) -> OutRunJob:
    if job_options is None:
        job_options = {}
    out_dir = OUTDIR / subdir
    my_logger.debug(type(datacube))
    datacube = datacube.filter_spatial(features)
    if run:
        job = datacube.create_job(
            title=title,
            description=description,
            out_format="netCDF",
            sample_by_feature=True,
            job_options=job_options,
        )
        job.start_job()
        print(f"Created job {job} with output dir {out_dir}")
    else:
        job = None
    return OutRunJob(
        job,
        out_dir,
        collection=datacube,
    )


def download_s2(
    connection,
    temporal_extent: list | None = None,
    max_cc: int = 100,
    year=None,
    features=FEATURES_VAL,
    tile=None,
    run=True,
) -> OutRunJob:
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    my_logger.debug(temporal_extent)
    sentinel2 = connection.load_collection(
        S2_COLLECTION,
        temporal_extent=temporal_extent,
        bands=S2_BANDS,
        max_cloud_cover=max_cc,
        properties={"provider:backend": lambda v: v == "vito"},
    )
    job_options = {
        "executor-memory": "6G",
        "executor-memoryOverhead": "12G",  # default 2G
        "executor-cores": 2,
        "task-cpus": 1,
        "executor-request-cores": "400m",
        "max-executors": "100",
        "driver-memory": "12G",
        "driver-memoryOverhead": "10G",
        "driver-cores": 5,
        "udf-dependency-archives": [],
        "logging-threshold": "info",
    }
    return run_job(
        datacube=sentinel2,
        title=f"Sentinel2_{tile}_{year}",
        description=f"Sentinel2_{tile}_L2A_{year}",
        subdir=sub_dir_name("S2", tile, year),
        features=features,
        job_options=job_options,
        run=run,
    )


def download_s1(
    connection,
    temporal_extent: list | None = None,
    collection_s2=None,
    orbit="ASCENDING",
    features=FEATURES_VAL,
    tile=None,
    year=None,
) -> OutRunJob:
    properties = {"sat:orbit_state": lambda od: od == orbit}
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    try:
        sentinel1 = connection.load_collection(
            "SENTINEL1_GRD",
            temporal_extent=temporal_extent,
            bands=["VV", "VH"],
            properties=properties,
        ).sar_backscatter(
            coefficient="gamma0-terrain",
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=True,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
            options=None,
        )
    except ValueError:
        sentinel1 = connection.load_collection(
            "SENTINEL1_GRD",
            temporal_extent=temporal_extent,
            bands=["VV", "VV+VH"],
            properties=properties,
        ).sar_backscatter(
            coefficient="gamma0-terrain",
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=True,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
            options=None,
        )

    if collection_s2 is not None:
        sentinel1 = sentinel1.resample_cube_spatial(
            collection_s2, method="cubic"
        )
    return run_job(
        datacube=sentinel1,
        title=f"Sentinel1_{orbit}_{tile}_{year}",
        description=f"Sentinel1_{orbit}_{tile}_{year}",
        subdir=sub_dir_name(f"S1_{orbit}", tile, year),
        features=features,
    )


def download_agora_per_band(
    connection,
    temporal_extent: list | None = None,
    collection_s2=None,
    features=FEATURES_VAL,
    tile=None,
    year=None,
    bands=None,
) -> OutRunJob:
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    if bands is None:
        bands = [
            "dewpoint-temperature",
            "precipitation-flux",
            "solar-radiation-flux",
            "temperature-max",
            "temperature-mean",
            "temperature-min",
            "vapour-pressure",
            "wind-speed",
        ]
    agera5 = connection.load_collection(
        "AGERA5",
        temporal_extent=temporal_extent,
        bands=bands,
    )
    if collection_s2 is not None:
        print("resample_spatial_cube in s2")
        agera5 = agera5.resample_cube_spatial(collection_s2, method="cubic")
    # job_options = {
    #     "executor-memory": "5G",
    #     "executor-memoryOverhead": "10G",  # default 2G
    #     "executor-cores": 1,
    #     "task-cpus": 1,
    #     "executor-request-cores": "400m",
    #     "max-executors": "100",
    #     "driver-memory": "12G",
    #     "driver-memoryOverhead": "10G",
    #     "driver-cores": 5,
    #     "udf-dependency-archives": [],
    #     "logging-threshold": "info",
    # }
    job_options = {
        "executor-memory": "5G",
        "executor-memoryOverhead": "10G",  # default 2G
        "executor-cores": 1,
        "task-cpus": 1,
        "executor-request-cores": "400m",
        "max-executors": "100",
        "driver-memory": "12G",
        "driver-memoryOverhead": "10G",
        "driver-cores": 5,
        "udf-dependency-archives": [],
        "logging-threshold": "info",
    }
    suffix = "_".join([D_AGERA5_BAND_NAME[b] for b in bands])
    return run_job(
        datacube=agera5,
        title=f"AGERA5_{tile}_{year}_{suffix}",
        description=f"AGERA5_{tile}_{year}_{suffix}",
        subdir=sub_dir_name(f"AGERA5_{suffix}", tile, year),
        features=features,
        job_options=job_options,
    )


def download_agora(
    connection,
    temporal_extent: list | None = None,
    collection_s2=None,
    features=FEATURES_VAL,
    tile=None,
    year=None,
) -> OutRunJob:
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    agera5 = connection.load_collection(
        "AGERA5",
        temporal_extent=temporal_extent,
        bands=[
            "dewpoint-temperature",
            "precipitation-flux",
            "solar-radiation-flux",
            "temperature-max",
            "temperature-mean",
            "temperature-min",
            "vapour-pressure",
            "wind-speed",
        ],
    )
    if collection_s2 is not None:
        print("resample_spatial_cube in s2")
        agera5 = agera5.resample_cube_spatial(collection_s2, method="cubic")
    job_options = {
        "executor-memory": "5G",
        "executor-memoryOverhead": "10G",  # default 2G
        "executor-cores": 1,
        "task-cpus": 1,
        "executor-request-cores": "400m",
        "max-executors": "100",
        "driver-memory": "12G",
        "driver-memoryOverhead": "10G",
        "driver-cores": 5,
        "udf-dependency-archives": [],
        "logging-threshold": "info",
    }
    return run_job(
        datacube=agera5,
        title=f"AGERA5_{tile}_{year}",
        description=f"AGERA5_{tile}_{year}",
        subdir=sub_dir_name("AGERA5", tile, year),
        features=features,
        job_options=job_options,
    )


def download_dem(
    connection, collection_s2=None, features=FEATURES_VAL, tile=None, year=None
) -> OutRunJob:
    dem = connection.load_collection(
        "COPERNICUS_30",
        bands=["DEM"],
    )
    if collection_s2 is not None:
        dem = dem.resample_cube_spatial(collection_s2, method="cubic")
    return run_job(
        dem,
        title=f"DEM{tile}",
        description=f"DEM_{tile}",
        subdir=sub_dir_name("DEM", tile, None),
        features=features,
    )


def sub_dir_name(
    suffix: str, tile: str | None = None, year: str | None = None
):
    path_subdir = ""
    if tile is not None:
        path_subdir += tile + "/"
        if year is not None:
            path_subdir += year + "/"
    path_subdir += suffix
    return path_subdir
