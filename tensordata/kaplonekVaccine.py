from os.path import dirname, join

import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def data():
    df = pd.read_csv(
        join(path_here, "tensordata/kaplonekVaccine2022/Luminex-functional-assay.csv"),
        delimiter=",",
        comment="#",
    )
    df = df.iloc[:, np.r_[1, 15:87]]
    df = pd.melt(
        df, id_vars=["Age.at.Presentation"], var_name="Measurement", value_name="Value"
    )
    df[["Receptor", "Antigen"]] = df["Measurement"].str.split("_", expand=True)
    df = df.drop(columns="Measurement")
    df = df.rename(columns={"Age.at.Presentation": "Subject"})
    return df.set_index(["Subject", "Antigen", "Receptor"]).to_xarray()["Value"]
