from openeo_mmdc.dataset.utils import build_dataset_info, merge_agera5_datasets


def test_merge_agera5_datasets():
    c_mmdc_df = build_dataset_info(
        path_dir="/media/dumeuri/DATA/Data/Iris/MMDC_OE",
        l_tile_s2=["30TYR"],
        list_modalities=[
            "dew_temp",
            "prec",
            "sol_rad",
            "temp_max",
            "temp_mean",
            "temp_min",
            "val_press",
            "wind_speed",
        ],
    )
    l_agera5_df = [
        c_mmdc_df.dew_temp,
        c_mmdc_df.temp_max,
        c_mmdc_df.temp_mean,
        c_mmdc_df.prec,
        c_mmdc_df.sol_rad,
        c_mmdc_df.temp_min,
        c_mmdc_df.sol_rad,
        c_mmdc_df.val_press,
    ]
    agera5_dataset = merge_agera5_datasets(l_agera5_df, item=0)
    print(agera5_dataset.data_vars)
