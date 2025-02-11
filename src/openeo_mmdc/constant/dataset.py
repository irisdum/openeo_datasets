from pathlib import Path

S2_COLLECTION = "SENTINEL2_L2A_SENTINELHUB"
S1_COLLECTION = "SENTINEL1_GRD"
S1_BANDS = ["VV", "VH"]
S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "SCL",
    "CLM",
    "sunAzimuthAngles",
    "sunZenithAngles",
    "viewAzimuthMean",
    "viewZenithMean",
    "dataMask",
]
S2_TILES_SHP = "/home/dumeuri/Documents/dataset/Sentinel-2-Shapefile-Index/sentinel_2_index_shapefile.shp"
L_TRAIN_TILES = [
    "30TXT",
    "31TBG",
    "31TDL",
    "31TEN",
    "32UMB",
    "32TPT",
    "32TPQ",
    "32UQD",
    "33TVM",
    "33UXR",
    "33TYJ",
    "34UEC",
    "34TFR",
]
L_VAL_TILES = [
    "30TYR",
    "31TEK",
    "32TNR",
    "32UPC",
    "33TXK",
    "34UDB",
]  # ["30TXT","30TYQ","30TYS","30UVU","31TBG","31TDJ","31TDL","31TFN","31TGJ","31UEP"]
TIMERANGE = ["2017-01-01", "2017-12-31"]
OUTDIR = Path("/home/dumeuri/Documents/dataset/MMDC_OE")

# FEATURES_VAL = "/home/dumeuri/Documents/dataset/datacubes/pretrain_val.geojson"

OUTDIR = Path("/home/ad/dumeuri/DeepChange/MMDC_OE/val")

FEATURES_TRAIN = sorted(
    Path("/home/ad/dumeuri/code/openeo_datasets/geojson").rglob(
        pattern="pretrain_train*.geojson"
    )
)
FEATURES_VAL = sorted(
    Path("/home/dumeuri/Documents/dataset/datacubes/geojson").rglob(
        pattern="pretrain_val*.geojson"
    )
)
D_AGERA5_BAND_NAME = dict(
    zip(
        [
            "dewpoint-temperature",
            "precipitation-flux",
            "solar-radiation-flux",
            "temperature-max",
            "temperature-mean",
            "temperature-min",
            "vapour-pressure",
            "wind-speed",
        ],
        [
            "DEW_TEMP",
            "PREC",
            "SOL_RAD",
            "TEMP_MAX",
            "TEMP_MEAN",
            "TEMP_MIN",
            "VAP_PRESS",
            "WIND_SPEED",
        ],
    )
)

REF_DATE = "2014-03-03"
TUPLE_REF_DATE = (2014, 3, 3)
BEG_TIME_SITS = "2017-01-01"
END_TIME_SITS = "2020-12-31"
