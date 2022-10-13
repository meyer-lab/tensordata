"""Import Zohar data, tensor formation, plotting raw data."""
from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd

path_here = dirname(dirname(__file__))

def data():

    df = pd.read_csv(join(path_here, "tensordata/zohar2020/ZoharCovData.csv"))
    df = df.dropna(subset=["days"]).reset_index(drop=True)
    
    params = df.iloc[:, 23:89].columns
    receptors = pd.unique([s.split("_")[0] for s in params])
    antigens = pd.unique([s.split("_")[1] for s in params])
    samples = pd.unique(df['sample_ID'])

    xdata = xr.DataArray(
        coords = {
            "Sample": samples,
            "Antigen": antigens,
            "Receptor": receptors
        },
        dims=("Sample", "Antigen", "Receptor")
    )

    for index, row in df.iterrows():
        for param in row.index[23:89]:
            R, Ag = param.split("_")
            xdata.loc[{"Sample": row["sample_ID"],
                       "Antigen": Ag,
                       "Receptor": R}] = df.loc[index, param]

    return xdata
    

