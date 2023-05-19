from os.path import join, dirname
from functools import lru_cache
import pandas as pd
import numpy as np

path_here = dirname(dirname(__file__))


@lru_cache(maxsize=1)
def data():
    df = pd.read_csv(join(path_here, "tensordata/atyeo2020/atyeo_covid.csv"), delimiter=",", comment="#")
    df = df.iloc[:22, np.r_[0, 13:64]]
    df = pd.melt(df, id_vars=['SampleID'], var_name='Measurement', value_name='Value')
    df[['Antigen', 'Receptor']] = df['Measurement'].str.split(' ', expand=True)
    df = df.drop(columns="Measurement")
    df = df.rename(columns={"SampleID": "Sample"})
    return df.set_index(["Sample", "Antigen", "Receptor"]).to_xarray()["Value"]