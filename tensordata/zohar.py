"""Import Zohar data, tensor formation, plotting raw data."""
from os.path import join, dirname
from functools import lru_cache
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))

@lru_cache(maxsize=1)
def data(subtract_baseline=False):
    df = pd.read_csv(join(path_here, "tensordata/zohar2020/ZoharCovData.csv"))
    baseline = df.loc[df['sample_ID'] == 'PBS'].values.squeeze()[23:89]
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["patient_ID"] = df["patient_ID"].astype('int32')
    df["group"] = pd.Categorical(df["group"])
    df = df.sort_values(by=["group", "days", "patient_ID"])
    df = df.dropna(subset=["days"])
    df = df.iloc[:,  np.r_[0, 23:89]]
    if subtract_baseline:
        df.iloc[:, 1:] -= baseline

    df = df.rename(columns={"sample_ID": "Sample"})
    df = pd.melt(df, id_vars='Sample', var_name='Measurement', value_name='Value')
    df[['Receptor', 'Antigen']] = df['Measurement'].str.split('_', expand=True)
    df = df.drop(columns="Measurement")
    return df.set_index(["Sample", "Antigen", "Receptor"]).to_xarray()["Value"]
