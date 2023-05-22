from dataclasses import dataclass
from pathlib import Path

import geopandas
import hydra
import numpy as np
import pandas as pd
import shapely

from openeo_mmdc.constant.dataset import L_TRAIN_TILES, S2_TILES_SHP

SEED = 5


def open_s2_fp(l_tile_name: list | None, file_path: str = S2_TILES_SHP):
    """

    Args:
        l_tile_name (): list of S2 tiles to select
        file_path (): shp files which contains s2 tiles geometry

    Returns:

    """
    assert Path(file_path).exists(), f"No file found at {file_path}"
    df = geopandas.read_file(file_path)

    if l_tile_name is None:
        l_tile_name = L_TRAIN_TILES
    df["cond"] = df.apply(lambda x: x.Name in l_tile_name, axis=1)
    sub_df = df[df["cond"]]

    assert len(sub_df) == len(
        l_tile_name
    ), "Not all tiles have been selected expected {} got {}".format(
        len(l_tile_name), len(sub_df)
    )
    return sub_df


def sample_random_bbox(
    polygon_geom: shapely.geometry,
    utm_crs,
    out_crs,
    box_size: int = 10240,
    num_boxes: int = 5,
    tile_name: str = "",
) -> geopandas.GeoDataFrame:
    """

    Args:
        polygon_geom (): the S2 tile footprint
        utm_crs (): the UTM crs adapted for this S2 tile
        out_crs (): the output crs
        box_size (): the roi size
        num_boxes (): the number of random roi to be created inside the polygon
        tile_name (): s2 tile name (written in the output  geopandas df)

    Returns:

    """
    # Generate random points within the polygon geometry
    assert box_size % 2 == 0, "Box size whould be divided by 2 {}".format(
        box_size
    )
    points = geopandas.points_from_xy(
        np.random.uniform(
            polygon_geom.bounds[0] + box_size / 2,
            polygon_geom.bounds[2] - box_size / 2,
            size=num_boxes,
        ),
        np.random.uniform(
            polygon_geom.bounds[1] + box_size / 2,
            polygon_geom.bounds[3] - box_size / 2,
            size=num_boxes,
        ),
        crs=utm_crs,
    )
    boxes = [
        shapely.geometry.box(
            point.x - box_size / 2,
            point.y - box_size / 2,
            point.x + box_size / 2,
            point.y + box_size / 2,
        )
        for point in points
    ]
    assert int(sum([round(b.area) for b in boxes])) == int(
        sum([round(box_size * box_size) for i in range(num_boxes)])
    ), "wrong dimension of the ROI should be {} not {}".format(
        [b.area for b in boxes], box_size * box_size
    )
    assert (
        int(sum([round(b.length) for b in boxes])) == box_size * num_boxes * 4
    ), "Wrong input dimn {} {} ".format(
        [b.length for b in boxes], box_size * num_boxes * 4
    )
    boxes_gdf = geopandas.GeoDataFrame(geometry=boxes, crs=utm_crs).to_crs(
        out_crs
    )
    boxes_gdf["Name"] = tile_name
    return boxes_gdf


def one_tile_bbox(
    tile_name: str,
    df: geopandas.GeoDataFrame,
    num_boxes: int = 10,
    out_crs: int | str = 4326,
    box_size=5120,
) -> geopandas.GeoDataFrame:
    """

    Args:
        box_size (): ROI size in meters
        out_crs (): output crs for the bbox
        tile_name (): S2 tile name
        df (): contains the footprint of each S2 tile name
        num_boxes (): number of ROI to create on each S2 tile
    Returns:
    a dataframe which contains the the num_boxes random roi
    selected on the tile indictaed by tile_name
    """
    tile_df = df[df["Name"] == tile_name]
    assert len(tile_df) > 0, f"No tile found ad {tile_name}"
    utm_crs = tile_df.estimate_utm_crs()
    utm_tile_df = tile_df.to_crs(utm_crs)
    bbox_df = sample_random_bbox(
        utm_tile_df["geometry"].iloc[0],
        utm_crs=utm_crs,
        out_crs=out_crs,
        tile_name=tile_name,
        num_boxes=num_boxes,
        box_size=box_size,
    )
    return bbox_df


@dataclass
class InputFP:
    dataset_type: str
    list_tile: list
    num_boxes: int
    box_size: int


@hydra.main(config_path="../../config/", config_name="generate_fp.yaml")
def main(config):
    input_fp = InputFP(
        list_tile=config.s2_tile,
        num_boxes=config.n_roi,
        box_size=config.roi_size,
        dataset_type=config.dataset_type,
    )
    list_tile = input_fp.list_tile
    df = open_s2_fp(l_tile_name=list_tile, file_path=config.path_s2_shp)
    print("orginal", df.crs)
    # print(L_TILES[0])
    print(len(df))
    l_bbox_df = []
    for tile in list_tile:
        l_bbox_df = [
            one_tile_bbox(
                tile_name=tile,
                df=df,
                num_boxes=input_fp.num_boxes,
                box_size=input_fp.box_size,
            )
        ]
        bbox_df = pd.concat(l_bbox_df)
        bbox_df.to_file(
            f"{config.ex_dir}/pretrain_{input_fp.dataset_type}_{tile}.geojson"
        )


if __name__ == "__main__":
    # train_input_fp = InputFP(
    #     "train", list_tile=L_TRAIN_TILES, num_boxes=10, box_size=5120
    # )
    # val_input_fp = InputFP(
    #     "val", list_tile=L_VAL_TILES, num_boxes=30, box_size=1280
    # )
    # input_fp = val_input_fp
    main()
