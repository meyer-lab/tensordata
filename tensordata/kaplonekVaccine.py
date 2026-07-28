import numpy as np

from .utils import melt_to_xarray, read_dataset_csv


def data():
    df = read_dataset_csv("kaplonekVaccine2022", "Luminex-functional-assay")
    df = df.iloc[:, np.r_[1, 15:87]]
    return melt_to_xarray(
        df,
        id_vars=["Age.at.Presentation"],
        split_into=["Receptor", "Antigen"],
        sep="_",
        index=["Subject", "Antigen", "Receptor"],
        rename={"Age.at.Presentation": "Subject"},
    )
