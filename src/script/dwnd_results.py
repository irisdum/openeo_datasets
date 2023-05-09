import openeo

if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    res = c.job("vito-j-e9178da12c0b4c9f8a586e071e871aa2").get_results()
    res.download_files(
        "/home/dumeuri/Documents/dataset/MMDC_OE/30TYR/2019/S1_ASCENDING"
    )
