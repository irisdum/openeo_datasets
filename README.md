# Setup

## Local env
`source build_conda_loc.sh`
## Env on hal cluster
`source build_conda_cluster.sh`
### Package
In this directory :
`pip install -e . `

# Run the code
To download data from openeo, we have designed two steps
1. Create a geojson file which contains ROI (polygons) where we want to download the SITS.
2. The creation and generation of the multimodal datacubes with openeo
3. The download of openeo datacubes
4. Clip the .nc files into the geosjon geomtries
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


To exploit --multirun option from hydra you can run :
`python generate_datacube.py year=2017,2018,2019,2020 --multirun ` WIth this command you download data for all
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
The download may not be perfect : we recommend checking for incorrectly downloaded file with : `find . -type f -size -50k`

### Requirements
To run this file needs the csv of the different batch_id, this csv is available :
[https://portal.terrascope.be/reporting](https://portal.terrascope.be/reporting)
(Do export to download)

## Clip the nc files into the geometry
We have noticed that the .nc files downloaded from openeo contains no data and does not perfectly
fit the polygon geometry given. The issue [https://discuss.eodc.eu/t/filter-spatial-create-datacubes-which-are-wider-than-the-input-polygons/581](https://discuss.eodc.eu/t/filter-spatial-create-datacubes-which-are-wider-than-the-input-polygons/581)
documents this case. Until this is fixed, a script to clip (crop) the .nc sits along the generated ROI is given.
The main script is `crop_data.py `and its configuration file is `crop.yaml`.

# Open the MMDC dataset as torch tensor
To open the SITS previously downloaded thanks to this script, you can use the `mmdc_sits` function defined in
`src/openeo_mmdc/dataset/load.py`.

# Conversion to pytorch Tensor
As openmf_files function is really slow, it slows the training process. Therefore we propose a script to convert the satellite image time series
in netcdf format to pytorch format (".pt"). The object saved is a class which has an attribute for each modalities.
For each modalities is stored :
- The reflectances (sits)
- The acquisition date as the difference in days with the reference_date
- (Optional) Acquisition mask
It is also possible to precise for s2 data the maximum cloud cover allowed in the patch.
