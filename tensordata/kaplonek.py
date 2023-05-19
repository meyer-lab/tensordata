from os.path import join, dirname
from functools import lru_cache
import xarray as xr
import pandas as pd
import numpy as np
from .util import split

path_here = dirname(dirname(__file__))

# import .read_csv
# organize data samplexantigenxreceptor
# find indices of each receptor and antigen, store


def load_file(name):
    """Return a requested data file."""
    data = pd.read_csv(
        join(path_here, "tensordata/kaplonek2021/" + name + ".csv"),
        delimiter=",",
        comment="#",
    )

    return data


@lru_cache(maxsize=1)
def SpaceX4D():
    data = load_file("SpaceX_Sero.Data")
    meta = load_file("SpaceX_meta.data")
    df = pd.concat([data, meta], join="outer", axis=1)
    df = df.drop(columns="Unnamed: 0")
    df = pd.melt(df, id_vars=['Pat.ID', 'time.point'], var_name='Measurement', value_name='Value')
    df[['Antigen', 'Receptor']] = df['Measurement'].str.split('-', expand=True)
    df = df.drop(columns="Measurement")
    df = df.rename(columns={"Pat.ID": "Subject", "time.point": "Time"})
    return df.set_index(["Subject", "Antigen", "Time", "Receptor"]).to_xarray()["Value"]


@lru_cache(maxsize=1)
def MGH4D():
    data = load_file("MGH_Sero.Neut.WHO124.log10")

    params = load_file("MGH_Features")
    antigens = pd.unique(params.values[:, 0].astype("str"))
    receptors = pd.unique(params.values[:, 1].astype("str"))

    samples = data.values[:, 0].astype("str")
    subjects = pd.unique([s.split("_")[0] for s in samples])
    days = pd.unique([s.split("_")[1] for s in samples])

    xdata = xr.DataArray(
        coords={
            "Subject": subjects,
            "Antigen": antigens,
            "Receptor": receptors,
            "Time": days,
        },
        dims=("Subject", "Antigen", "Receptor", "Time"),
    )

    for index, row in data.iterrows():
        for param in row.index[1:91]:
            Ag, R = split(param, ".", -1)
            sub, day = row["Unnamed: 0"].split("_")
            xdata.loc[
                {"Subject": sub, "Time": day, "Antigen": Ag, "Receptor": R}
            ] = data.loc[index, param]

    func_feats = data.columns[-4:]

    func_xdata = xr.DataArray(
        coords={
            "Subject": subjects,
            "Feature": func_feats,
            "Time": days,
        },
        dims=("Subject", "Feature", "Time"),
    )

    for index, row in data.iterrows():
        sub, day = row["Unnamed: 0"].split("_")
        func_xdata.loc[{"Subject": sub, "Time": day}] = row[-4:]

    return xr.Dataset(
        data_vars={"Serology": xdata, "Functional": func_xdata},
        coords={
            "Subject": subjects,
            "Feature": func_feats,
            "Antigen": antigens,
            "Receptor": receptors,
            "Time": days,
        },
    )

def MGH4D2():
    data = load_file("MGH_Sero.Neut.WHO124.log10")
    data = data.rename(columns={"Unnamed: 0": "Sample"})
    data[['Subject', 'Time']] = data['Sample'].str.split('_', expand=True)
    params = load_file("MGH_Features")

    func = data.iloc[:, 91:]
    data = data.iloc[:, np.r_[1:91, 95:97]]
    data = pd.melt(data, id_vars=['Subject', 'Time'], var_name='Names', value_name='Serology')
    data = data.join(params.set_index('Names'), on="Names").drop(columns="Names")
    dx = data.set_index(['Subject', 'Time', "Antigen", "Receptor"]).to_xarray()

    func = pd.melt(func, id_vars=['Subject', 'Time'], var_name='Feature', value_name='Functional')
    dx["Functional"] = func.set_index(['Subject', 'Time', "Feature"]).to_xarray()["Functional"]
    return dx
