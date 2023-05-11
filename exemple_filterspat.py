import json
from pathlib import Path

import openeo

if __name__ == "__main__":
    temporal_extent = TIMERANGE = ["2020-01-01", "2020-01-27"]
    OUTDIR = Path("/home/dumeuri/Documents/dataset/MMDC_OE")
    S2_BANDS = ["B02", "B03", "B04"]
    if temporal_extent is None:
        temporal_extent = TIMERANGE

    connection = openeo.connect("openeo.cloud")
    connection.authenticate_oidc()
    sentinel2 = connection.load_collection(
        "SENTINEL2_L2A_SENTINELHUB",
        temporal_extent=temporal_extent,
        bands=S2_BANDS,
    )

    features_path = "/home/dumeuri/Documents/dataset/datacubes/geojson/pretrain_val_31TEK.geojson"
    with open(features_path) as feat:
        print(feat)
        features = json.load(feat)
    # sentinel2=sentinel2.resample_spatial(resolution=0,projection="EPSG:4326")
    datacube = sentinel2.filter_spatial(features)
    datacube = datacube.resample_spatial(resolution=0.0001)
    datacube = datacube.filter_spatial(features)
    # datacube=datacube.mask_polygon(mask=features_path,replacement = NA)
    job = datacube.create_job(
        title="S2_minimal_T31TEK",
        description="S2_minimal_T31TEK_resample_res",
        out_format="netCDF",
        sample_by_feature=True,
    )
    job.start_job()
