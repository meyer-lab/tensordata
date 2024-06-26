from os.path import dirname, join

import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def data():
    df = pd.read_csv(
        join(path_here, "tensordata/chung2021/fig6.csv"), delimiter=",", comment="#"
    )
    df.columns = df.columns.str.replace("Pan IgG", "PanIgG")
    df = df.iloc[:, np.r_[0, 5:229]]
    df = pd.melt(df, id_vars=["Patient"], var_name="Measurement", value_name="Value")
    df[["Antigen", "Receptor"]] = df["Measurement"].str.rsplit(" ", n=1, expand=True)
    df = df.drop(columns="Measurement")
    df = df.rename(columns={"Patient": "Subject"})
    return df.set_index(["Subject", "Antigen", "Receptor"]).to_xarray()["Value"]
