import numpy as np
import pandas as pd

from .utils import melt_to_xarray, read_dataset_csv


def data(subtract_baseline=False):
    df = read_dataset_csv("zohar2020", "ZoharCovData")
    baseline = df.loc[df["sample_ID"] == "PBS"].iloc[:, 23:89].values.squeeze()
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["patient_ID"] = df["patient_ID"].astype("int32")
    df["group"] = pd.Categorical(df["group"])
    df = df.sort_values(by=["group", "days", "patient_ID"])
    df = df.dropna(subset=["days"])
    df = df.iloc[:, np.r_[0, 23:89]]
    if subtract_baseline:
        df.iloc[:, 1:] -= baseline

    df = df.rename(columns={"sample_ID": "Sample"})
    return melt_to_xarray(
        df,
        id_vars=["Sample"],
        split_into=["Receptor", "Antigen"],
        sep="_",
        index=["Sample", "Antigen", "Receptor"],
    )
