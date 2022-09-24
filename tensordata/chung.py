from os.path import join, dirname
import pandas as pd
import xarray as xr

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/chung2021/" + name + ".csv"), delimiter=",", comment="#")

    return data
    
def data():
    data = load_file("fig6")

    params = data.iloc[:, 21:].columns
    params = [s.replace("\n", "") for s in params]
    
    antigens = pd.unique([split(s, " ", -1)[0] for s in params])
    receptors = pd.unique([split(s, " ", -1)[1] for s in params])
    subjects = pd.unique(data.loc[:, "Patient"])

    xdata = xr.DataArray(
        coords = {
            "Subject": subjects,
            "Antigen": antigens,
            "Receptor": receptors
        },
        dims=("Subject", "Antigen", "Receptor")
    )

    for index, row in data.iterrows():
        for param in row.index[21:]:
            Ag, R = split(param, " ", -1)
            xdata.loc[{"Subject": row["Patient"],
                       "Antigen": Ag,
                       "Receptor": R}] = data.loc[index, param]

    return xdata


def split(str, sep, pos):
    str = str.split(sep)
    return sep.join(str[:pos]), sep.join(str[pos:])