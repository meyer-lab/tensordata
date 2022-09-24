from os.path import join, dirname
import pandas as pd
import xarray as xr

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/kaplonekVaccine2022/" + name + ".csv"), delimiter=",", comment="#")

    return data


def data():
    data = load_file("Luminex-functional-assay")

    params = data.iloc[:, 15:].columns
    receptors = pd.unique([s.split("_")[0] for s in params])
    antigens = pd.unique([s.split("_")[1] for s in params])
    subjects = pd.unique(data.loc[:, "Age.at.Presentation"])

    xdata = xr.DataArray(
        coords = {
            "Subject": subjects,
            "Antigen": antigens,
            "Receptor": receptors
        },
        dims=("Subject", "Antigen", "Receptor")
    )

    for index, row in data.iterrows():
        for param in row.index[15:]:
            R, Ag = param.split("_")
            xdata.loc[{"Subject": row["Age.at.Presentation"],
                       "Antigen": Ag,
                       "Receptor": R}] = data.loc[index, param]

    return xdata




