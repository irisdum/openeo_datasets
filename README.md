# Installation
`pip install -e . `
`pre-commit install `
## Environment
# Run the code
To download data from openeo, we have designed two steps
1. Create a geojson file which contains ROI (polygons) where we want to download the SITS.
2. The creation and generation of the multimodal datacubes with openeo
3. The download of openeo datacubes
## Create ROIs geojson files
The script located at `src/script/generate_fp.py` enables downloading the image.
To run, this script requires the `sentinel_2_index_shapefile.shp`.
To define the ROI size, the number of ROI per S2 tiles, change the config file` generate_fp.yaml`
This script will create a geojson per S2 tiles.

## Create datacubes in openeo
The script localted at `src/script/generate_datacube.py` creates datacubes in openeo. This script download multimodal
data for one year on one geojson file generated at the previous step (whic corresponds to ROIs in one S2 tiles).
Due to memory issue on the vito cluster, each of the weather variables (from AGERA5 collection are downloaded separately).
This script create datacubes resampled at the S2 resolution for the following collections:
- Sentinel 2
- Sentinel 1 Ascending
- Sentinel 1 Descending
- DEM
- AGERA5
### Requirements:
To run this script needs:
- A geojson fil with the ROIs
- The definition of a time range (not recommmended to geo over one year for memory issue on the vito cluster)
Change the config file `generate_datacube.yaml`
### See jobs in vito batch cluster
To see if the batch job created are running, connect to [https://editor.openeo.cloud/](https://editor.openeo.cloud/).

## Download the datacubes
The script located at `src/script/dwnd_cluster.py` enables downloading the data locally.
Modify paths defined in `config/dwnd.yaml `.
### Requirements
To run this file needs the csv of the different batch_id, this csv is available :
[https://portal.terrascope.be/reporting](https://portal.terrascope.be/reporting)
(Do export to download)
