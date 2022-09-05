"""Import Zohar data, tensor formation, plotting raw data."""
from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd

path_here = dirname(dirname(__file__))

def pbsSubtractOriginal():
    """ Paper Background subtract, will keep all rows for any confusing result. """
    Cov = pd.read_csv(join(path_here, "tensordata/zohar2020/ZoharCovData.csv"), index_col=0)
    # 23 (0-> 23) is the start of IgG1_S
    Demographics = Cov.iloc[:, 0:23]
    Serology = Cov.iloc[:, 23::]
    Serology -= Serology.loc["PBS"].values.squeeze()
    df = pd.concat([Demographics, Serology], axis=1)
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["week"] = np.array(df["days"] // 7 + 1.0, dtype=int)
    df["patient_ID"] = df["patient_ID"].astype('int32')
    df["group"] = pd.Categorical(df["group"], ["Negative", "Mild", "Moderate", "Severe", "Deceased"])
    df = df.sort_values(by=["group", "days", "patient_ID"])
    df = df.reset_index()
    # Get rid of any data that doesn't have a time component (i.e. "nan" for day)
    df = df.dropna(subset=["days"]).reset_index(drop=True)
    return df


def data(xarray = False):
    df = pbsSubtractOriginal()
    
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
        dims=("Subject", "Antigen", "Receptor")
    )

    for index, row in df.iterrows():
        for param in row.index[23:89]:
            R, Ag = param.split("_")
            xdata.loc[{"Sample": row["sample_ID"],
                       "Antigen": Ag,
                       "Receptor": R}] = df.loc[index, param]

    return xdata
    

