AGERA_DATA = [
    "DEW_TEMP",
    "PREC",
    "SOL_RAD",
    "TEMP_MAX",
    "TEMP_MEAN",
    "TEMP_MIN",
    "VAP_PRESS",
    "WIND_SPEED",
]
L_MOD_MMDC = [
    "s2",
    "s1_asc",
    "s1_desc",
    "dem",
    "dew_temp",
    "prec",
    "sol_rad",
    "temp_max",
    "temp_mean",
    "temp_min",
    "val_press",
    "wind_speed",
]
D_MODALITY = dict(
    zip(
        L_MOD_MMDC,
        ["Sentinel2", "Sentinel1_ASCENDING", "Sentinel1_DESCENDING", "DEM"]
        + AGERA_DATA,
    )
)

S2_TILE = ["31TEK"]
FORMAT_SITS = "clip.nc"

S2_BAND = [
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
]

CLD_MASK = "CLM"
CLD_MASK_BAND = ["CLM", "SCL"]
LOAD_VARIABLE = S2_BAND + [CLD_MASK]
S1_BAND = ["VH", "VV", "local_incidence_angle"]
AGERA_BAND = [
    "dewpoint-temperature",
    "precipitation-flux",
    "solar-radiation-flux",
    "temperature-max",
    "temperature-mean",
    "temperature-min",
    "vapour-pressure",
    "wind-speed",
]
