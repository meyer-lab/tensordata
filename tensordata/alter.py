from os.path import dirname, join

import numpy as np
import pandas as pd
import xarray as xr

path_here = dirname(dirname(__file__))


def load_file(name):
    """Return a requested data file."""
    data = pd.read_csv(
        join(path_here, "tensordata/alter2018/" + name + ".csv"),
        delimiter=",",
        comment="#",
    )
    return data


def data():
    # Get axes
    subjects = load_file("meta-subjects")["subject"].to_list()
    detections = load_file("meta-detections")["detection"].to_list()
    antigen = load_file("meta-antigens")["antigen"].to_list()

    # Assemble Fc measurements
    cube = xr.DataArray(
        data=np.nan,
        coords={"Sample": subjects, "Receptor": detections, "Antigen": antigen},
        dims=["Sample", "Receptor", "Antigen"],
    )
    cube = cube.stack(index=("Sample", "Receptor", "Antigen")).to_pandas().reset_index()
    cube["Measurement"] = cube["Receptor"] + "." + cube["Antigen"]

    df = load_file("data-luminex")
    df = pd.merge(df, load_file("data-luminex-igg"), on="subject")
    df = df.rename(columns={"subject": "Sample"})
    df = pd.melt(df, id_vars=["Sample"], var_name="Measurement", value_name="Fc")
    df = pd.merge(cube, df, on=["Sample", "Measurement"])
    df = df.drop(columns=[0, "Measurement"])
    df = df.set_index(["Sample", "Receptor", "Antigen"]).to_xarray()
    # raise columns with negative values by the smallest so all is positive
    df["Fc"] -= (df["Fc"] < 0.0).any(dim="Sample") * df["Fc"].min(dim="Sample")

    # Assemble glycan info (check, all good!)
    glycan = load_file("data-glycan-gp120")
    glycan = glycan.rename(columns={"subject": "Sample"})
    glycan = pd.melt(glycan, id_vars=["Sample"], var_name="Glycan", value_name="gp120")
    df["gp120"] = glycan.set_index(["Sample", "Glycan"]).to_xarray()["gp120"]

    # Import function
    functional = load_file("data-function")
    functional = functional.rename(
        columns={"subject": "Sample", "IFNy": "IFNγ", "MIP1b": "MIP1β"}
    )
    functional = pd.melt(
        functional, id_vars=["Sample"], var_name="Function", value_name="Functional"
    )
    df["Functional"] = functional.set_index(["Sample", "Function"]).to_xarray()[
        "Functional"
    ]
    return df
