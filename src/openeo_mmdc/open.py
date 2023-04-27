import pandas as pd


def open_job_df(path_csv: str):
    df = pd.read_csv(path_csv)
    print(df.columns)
    return ["vito-" + jobid for jobid in df["jobId"].tolist()]


if __name__ == "__main__":
    print(
        open_job_df(
            "/home/dumeuri/Téléchargements/reporting_export_20230427(1).csv"
        )
    )
