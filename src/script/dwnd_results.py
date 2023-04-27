import openeo


def extract_year(dict_metadata: dict):
    pass


def extract_tile(dict_metadat: dict):
    pass


if __name__ == "__main__":
    c = openeo.connect("openeo.cloud")
    c.authenticate_oidc()
    job = c.job("vito-j-dbdccbcd427344c28b9842d5b38b2c3b")

    res = job.get_results()
    dict_metadata = res.get_metadata()

    print(res.get_metadata()["properties"]['card4l:processing_chain']
          ["process_graph"]['loadcollection1']["arguments"]["temporal_extent"])
    res.download_files("/home/ad/dumeuri/DeepChange/MMDC_OE/val/30TYR/2017/S2")
