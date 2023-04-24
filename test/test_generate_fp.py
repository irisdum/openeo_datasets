from script.generate_fp import open_s2_fp


def test_open_s2_fp():
    df = open_s2_fp(l_tile_name=None)
    print(df["geometry"].iloc[0].bounds)
    print(df.crs)
    bounds = df["geometry"].iloc[0].bounds
    print(type(bounds))
    sub_df = df.iloc[[0]]
    utm_crs = sub_df.estimate_utm_crs()
    utm_df = sub_df.to_crs(utm_crs)
    print(utm_df["geometry"].iloc[0].bounds)
