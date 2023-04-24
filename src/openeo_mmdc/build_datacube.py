from dataclasses import dataclass
from pathlib import Path

from openeo import BatchJob, DataCube

from openeo_mmdc.constant.dataset import (
    FEATURES,
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
    datacube: DataCube, title: str, description: str, subdir: str
) -> OutRunJob:
    out_dir = OUTDIR / subdir
    print(type(datacube))
    job = datacube.filter_spatial(
        FEATURES,
    ).create_job(
        title=title,
        description=description,
        out_format="netCDF",
        sample_by_feature=True,
    )
    job.start_job()
    print(f"Created job {job} with output dir {out_dir}")

    return OutRunJob(job, out_dir, collection=datacube)


def download_s2(
    connection, temporal_extent: list | None = None, max_cc=50
) -> OutRunJob:
    if temporal_extent is None:
        temporal_extent = TIMERANGE
    sentinel2 = connection.load_collection(
        S2_COLLECTION,
        temporal_extent=temporal_extent,
        bands=S2_BANDS,
        max_cloud_cover=max_cc,
    )
    return run_job(
        datacube=sentinel2,
        title="Sentinel2",
        description="Sentinel-2 L2A bands",
        subdir="S2",
    )


def download_s1(
    connection,
    temporal_extent: list | None = None,
    collection_s2=None,
    orbit="ASCENDING",
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
        sentinel1 = sentinel1.resample_cube_spatial(
            collection_s2, method="cubic"
        )
    return run_job(
        datacube=sentinel1,
        title=f"Sentinel1_{orbit}",
        description=f"Sentinel-1 VV, VH, orbit {orbit}",
        subdir=f"S1_{orbit}",
    )


def download_agora(
    connection, temporal_extent: list | None = None, collection_s2=None
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
        title="AGERA5",
        description="AGERA-5",
        subdir="AGERA5",
    )


def download_dem(connection, collection_s2=None) -> OutRunJob:
    dem = connection.load_collection(
        "COPERNICUS_30",
        bands=["DEM"],
    )
    if collection_s2 is not None:
        dem = dem.resample_cube_spatial(collection_s2, method="cubic")
    return run_job(dem, title="DEM", description="DEM", subdir="DEM")
