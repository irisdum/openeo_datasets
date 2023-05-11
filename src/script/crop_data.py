from pathlib import Path

import geopandas
import hydra
import rasterio
import rioxarray


@hydra.main(config_path="../../config/", config_name="crop.yaml")
def main(config):
    path_dir_geojson = Path(config.path_geojson)
    gene_path = path_dir_geojson.glob(f"*{config.s2_tile}.geojson")
    path_geojson = [p for p in gene_path][0]
    gdf = geopandas.read_file(path_geojson)
    utm_crs = gdf.estimate_utm_crs()
    utm_tile_df = gdf.to_crs(utm_crs)
    for index, row in utm_tile_df.iterrows():
        assert Path(config.path_global_dir, config.s2_tile).exists()
        gene_path_image = Path(config.path_global_dir, config.s2_tile).glob(
            f"**/openEO_{index}.nc"
        )
        l_path_image = [p for p in gene_path_image]
        assert len(l_path_image) == 4 * 11 + 1, print(len(l_path_image))
        for path_image in l_path_image:  # TODO maybe parallelize with dask
            ex_path = Path(path_image.parent, f"openEO_{index}_clip.nc")
            try:
                if not ex_path.exists():
                    with rioxarray.open_rasterio(
                        path_image, decode_times=False
                    ) as xds:
                        print(path_image)
                        clip = xds.rio.clip([row["geometry"]])
                        clip.to_netcdf(path=ex_path)
            except rasterio.errors.RasterioIOError:
                print(f"Not able to open {path_image}")


if __name__ == "__main__":
    main()
