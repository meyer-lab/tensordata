from os.path import join, dirname
import pandas as pd
import xarray as xr
import numpy as np
from .util import split 

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/chung2021/" + name + ".csv"), delimiter=",", comment="#")

    return data
    
def data():
    data = load_file("fig6")

    pan_params = data.iloc[:, 5:21].columns
    params = data.iloc[:, 21:].columns

    antigens = pd.unique([split(s, " ", -1)[0] for s in params])
    pan_receptors = pd.unique([split(s, " ", -2)[1] for s in pan_params])
    receptors = pd.unique([split(s, " ", -1)[1] for s in params])
    subjects = pd.unique(data.loc[:, "Patient"])

    xdata = xr.DataArray(
        coords = {
            "Subject": subjects,
            "Antigen": antigens,
            "Receptor": np.concatenate((pan_receptors, receptors))
        },
        dims=("Subject", "Antigen", "Receptor")
    )

    for index, row in data.iterrows():
        for pan_param in row.index[5:21]:
            Ag, R = split(pan_param, " ", -2)
            xdata.loc[{"Subject": row["Patient"],
                       "Antigen": Ag,
                       "Receptor": R}] = data.loc[index, pan_param]
        for param in row.index[21:]:
            Ag, R = split(param, " ", -1)
            xdata.loc[{"Subject": row["Patient"],
                       "Antigen": Ag,
                       "Receptor": R}] = data.loc[index, param]

    return xdata

