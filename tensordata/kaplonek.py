from os.path import join, dirname
from functools import lru_cache
import xarray as xr
import pandas as pd
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
    data = pd.concat([data, meta], join="outer", axis=1)

    params = data.iloc[:, 1:85].columns
    antigens = pd.unique([s.split("-")[0] for s in params])
    antibodies = pd.unique([s.split("-")[1] for s in params])
    patients = pd.unique(data.loc[:, "Pat.ID"])
    days = pd.unique(data.loc[:, "time.point"])

    xdata = xr.DataArray(
        coords={
            "Subject": patients,
            "Antigen": antigens,
            "Receptor": antibodies,
            "Time": days,
        },
        dims=("Subject", "Antigen", "Receptor", "Time"),
    )

    for index, row in data.iterrows():
        for param in row.index[1:85]:
            Ag, Ab = param.split("-")
            xdata.loc[
                {
                    "Subject": row["Pat.ID"],
                    "Time": row["time.point"],
                    "Antigen": Ag,
                    "Receptor": Ab,
                }
            ] = data.loc[index, param]

    return xdata


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
