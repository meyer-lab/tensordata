from os.path import join, dirname
from functools import lru_cache
import xarray as xr
import pandas as pd

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/atyeo2020/" + name + ".csv"), delimiter=",", comment="#")

    return data


@lru_cache(maxsize=1)
def data():
    data = load_file("atyeo_covid")
    data = data.iloc[:22, :]

    params = data.iloc[:, 13:43].columns
    antigens = pd.unique([s.split(" ")[0] for s in params])
    receptors = pd.unique([s.split(" ")[1] for s in params])
    subjects = pd.unique(data.loc[:, "SampleID"])

    xdata = xr.DataArray(
        coords = {
            "Subject": subjects,
            "Antigen": antigens,
            "Receptor": receptors
        },
        dims=("Subject", "Antigen", "Receptor")
    )

    for index, row in data.iterrows():
        for param in row.index[13:43]:
            Ag, R = param.split(" ")
            xdata.loc[{"Subject": row["SampleID"],
                       "Antigen": Ag,
                       "Receptor": R}] = data.loc[index, param]

    return xdata