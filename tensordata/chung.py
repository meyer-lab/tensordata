import numpy as np

from .utils import melt_to_xarray, read_dataset_csv


def data():
    df = read_dataset_csv("chung2021", "fig6")
    df.columns = df.columns.str.replace("Pan IgG", "PanIgG")
    df = df.iloc[:, np.r_[0, 5:229]]
    return melt_to_xarray(
        df,
        id_vars=["Patient"],
        split_into=["Antigen", "Receptor"],
        sep=" ",
        rsplit=True,
        maxsplit=1,
        index=["Subject", "Antigen", "Receptor"],
        rename={"Patient": "Subject"},
    )
