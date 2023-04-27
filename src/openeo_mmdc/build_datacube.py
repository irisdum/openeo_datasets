from dataclasses import dataclass
from pathlib import Path

from openeo import BatchJob, DataCube

from openeo_mmdc.constant.dataset import (
    FEATURES_VAL,
    OUTDIR,
    S2_BANDS,
    S2_COLLECTION,
    TIMERANGE,
)


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
) -> OutRunJob:
    out_dir = OUTDIR / subdir
    print(type(datacube))
    job = datacube.filter_spatial(features).create_job(
        title=title,
        description=description,
        out_format="netCDF",
        sample_by_feature=True,
    )
    job.start_job()
    print(f"Created job {job} with output dir {out_dir}")

    return OutRunJob(job, out_dir, collection=datacube)


def download_s2(
    connection,
    temporal_extent: list | None = None,
    max_cc=50,
    year=None,
    features=FEATURES_VAL,
    tile=None,
) -> OutRunJob:
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    print(temporal_extent)
    sentinel2 = connection.load_collection(S2_COLLECTION,
                                           temporal_extent=temporal_extent,
                                           bands=S2_BANDS)
    return run_job(
        datacube=sentinel2,
        title=f"Sentinel2_{tile}",
        description=f"Sentinel-2 {tile} L2A bands {year}",
        subdir=sub_dir_name("S2", tile, year),
        features=features,
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

    if collection_s2 is not None:
        sentinel1 = sentinel1.resample_cube_spatial(collection_s2,
                                                    method="cubic")
    return run_job(
        datacube=sentinel1,
        title=f"Sentinel1_{orbit}_{tile}",
        description=f"Sentinel-1 VV, VH, orbit {orbit} {tile} {year}",
        subdir=sub_dir_name(f"S1_{orbit}", tile, year),
        features=features,
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
        agera5 = agera5.resample_cube_spatial(collection_s2, method="cubic")
    return run_job(
        datacube=agera5,
        title=f"AGERA5_{tile}",
        description=f"AGERA-5{tile} {year}",
        subdir=sub_dir_name("AGERA5", tile, year),
        features=features,
    )


def download_dem(connection,
                 collection_s2=None,
                 features=FEATURES_VAL,
                 tile=None,
                 year=None) -> OutRunJob:
    dem = connection.load_collection(
        "COPERNICUS_30",
        bands=["DEM"],
    )
    if collection_s2 is not None:
        dem = dem.resample_cube_spatial(collection_s2, method="cubic")
    return run_job(
        dem,
        title=f"DEM{tile}",
        description=f"DEM_{tile}_{year}",
        subdir=sub_dir_name("DEM", tile, year),
        features=features,
    )


def sub_dir_name(suffix: str,
                 tile: str | None = None,
                 year: str | None = None):
    path_subdir = suffix
    if tile is not None:
        path_subdir += "/" + tile
        if year is not None:
            path_subdir += "/" + year
    return path_subdir
