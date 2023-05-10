import json
from pathlib import Path

import openeo

if __name__ == "__main__":
    temporal_extent = TIMERANGE = ["2020-01-01", "2020-02-27"]
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

    features = "/home/dumeuri/Documents/dataset/datacubes/geojson/pretrain_val_31TEK.geojson"
    with open(features) as feat:
        print(feat)
        features = json.load(feat)
    datacube = sentinel2.filter_spatial(features)
    job = datacube.create_job(
        title="S2_minimal_T31TEK",
        description="S2_minimal_T31TEK",
        out_format="netCDF",
        sample_by_feature=True,
    )
    job.start_job()
