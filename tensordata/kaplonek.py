from os.path import dirname, join

import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def load_file(name):
    """Return a requested data file."""
    data = pd.read_csv(
        join(path_here, "tensordata/kaplonek2021/" + name + ".csv"),
        delimiter=",",
        comment="#",
    )
    return data


def SpaceX4D():
    data = load_file("SpaceX_Sero.Data")
    meta = load_file("SpaceX_meta.data")
    df = pd.concat([data, meta], join="outer", axis=1)
    df = df.drop(columns="Unnamed: 0")
    df = pd.melt(
        df, id_vars=["Pat.ID", "time.point"], var_name="Measurement", value_name="Value"
    )
    df[["Antigen", "Receptor"]] = df["Measurement"].str.split("-", expand=True)
    df = df.drop(columns="Measurement")
    df = df.rename(columns={"Pat.ID": "Subject", "time.point": "Time"})
    return df.set_index(["Subject", "Antigen", "Receptor", "Time"]).to_xarray()["Value"]


def MGH4D():
    data = load_file("MGH_Sero.Neut.WHO124.log10")
    data = data.rename(columns={"Unnamed: 0": "Sample"})
    data[["Subject", "Time"]] = data["Sample"].str.split("_", expand=True)
    params = load_file("MGH_Features")

    func = data.iloc[:, 91:]
    data = data.iloc[:, np.r_[1:91, 95:97]]
    data = pd.melt(
        data, id_vars=["Subject", "Time"], var_name="Names", value_name="Serology"
    )
    data = data.join(params.set_index("Names"), on="Names").drop(columns="Names")
    dx = data.set_index(["Subject", "Antigen", "Receptor", "Time"]).to_xarray()

    func = pd.melt(
        func, id_vars=["Subject", "Time"], var_name="Feature", value_name="Functional"
    )
    dx["Functional"] = func.set_index(["Subject", "Feature", "Time"]).to_xarray()[
        "Functional"
    ]
    return dx
