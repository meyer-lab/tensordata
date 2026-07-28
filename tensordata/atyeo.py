import numpy as np

from .utils import melt_to_xarray, read_dataset_csv


def data():
    df = read_dataset_csv("atyeo2020", "atyeo_covid")
    df = df.iloc[:22, np.r_[0, 13:64]]
    return melt_to_xarray(
        df,
        id_vars=["SampleID"],
        split_into=["Antigen", "Receptor"],
        sep=" ",
        index=["Sample", "Antigen", "Receptor"],
        rename={"SampleID": "Sample"},
    )
