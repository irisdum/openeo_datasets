import logging
from pathlib import Path

import dask
import geopandas
import hydra
import rasterio
import rioxarray

my_logger = logging.getLogger(__name__)


def clip_sits(path_image, ex_path, max_cc, row) -> str:
    try:
        if not ex_path.exists():
            with rioxarray.open_rasterio(
                path_image, decode_times=False, chunks="auto", parallel=True
            ) as xds:
                print(path_image)

                clip = xds.rio.clip([row["geometry"]])
                print(path_image.absolute())
                if (max_cc is not None) and (
                    "Sentinel2" in str(path_image.absolute())
                ):
                    my_logger.info(f"Select only image with CC below {max_cc}")
                    max_pix_cc = clip.sizes["x"] * clip.sizes["y"] * max_cc
                    ccp = clip[["CLM"]].sum(dim=["x", "y"])
                    ccp = ccp.compute()
                    clip = clip.sel(
                        t=ccp.where(ccp["CLM"] < max_pix_cc, drop=True)["t"]
                    )

                clip.to_netcdf(path=ex_path)
        return ex_path
    except rasterio.errors.RasterioIOError:
        print(f"Not able to open {path_image}")
        return ""


@hydra.main(config_path="../../config/", config_name="crop.yaml")
def main(config):
    path_dir_geojson = Path(config.path_geojson)
    print(f"{path_dir_geojson},{config.s2_tile}")
    gene_path = path_dir_geojson.glob(f"*{config.s2_tile}*.geojson")
    path_geojson = [p for p in gene_path][0]
    gdf = geopandas.read_file(path_geojson)
    utm_crs = gdf.estimate_utm_crs()
    utm_tile_df = gdf.to_crs(utm_crs)
    max_cc = config.max_cc
    l_ex_path = []
    for index, row in utm_tile_df.iterrows():
        assert Path(
            config.path_global_dir, config.s2_tile
        ).exists(), (
            f"{Path(config.path_global_dir, config.s2_tile)} not found "
        )
        # print(f"{config.path_global_dir}, {config.s2_tile}")
        gene_path_image = Path(config.path_global_dir, config.s2_tile).glob(
            f"**/openEO_{index}.nc"
        )
        l_path_image = [p for p in gene_path_image]
        # assert len(l_path_image) == 4 * 11 + 1, print(len(l_path_image))

        for path_image in l_path_image:  # TODO maybe parallelize with dask
            ex_path = Path(
                path_image.parent,
                f"openEO_{index}_clip.nc",
            )
            l_ex_path += [
                dask.delayed(clip_sits)(path_image, ex_path, max_cc, row)
            ]
    with dask.config.set(scheduler="processes"):
        dask.compute(*l_ex_path)


if __name__ == "__main__":
    main()
