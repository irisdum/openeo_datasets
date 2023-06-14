import openeo


def extract_year(dict_metadata: dict):
    pass


def extract_tile(dict_metadat: dict):
    pass


if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    job = c.job("vito-j-7bc22eb88feb403db0866a1e4e948301")

    res = job.get_results()
    dict_metadata = res.get_metadata()

    print(
        res.get_metadata()["properties"]["card4l:processing_chain"][
            "process_graph"
        ]["loadcollection1"]["arguments"]["temporal_extent"]
    )
    res.download_files(
        "/media/dumeuri/DATA/Data/Iris/MMDC_OE/32TNR/2020/Sentinel2_32TNR_2020"
    )
