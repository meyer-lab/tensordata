from locale import D_FMT
from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xa

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/chung2021/" + name + ".csv"), delimiter=",", comment="#")

    return data


def importFig1():
    data = load_file("fig1")
    xdf = data.to_xarray()
    return xdf.to_array()




    


