import openeo

if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    res = c.job("vito-j-a970d643986c4f0e988a67474c9c3e13").get_results()
    res.download_files(
        "/home/dumeuri/Documents/dataset/datacubes/multimodal/s1_des"
    )
