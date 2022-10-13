from .kaplonek import MGH4D, SpaceX4D
from .zohar import data as Zohar
from tensorly.regression.cp_plsr import *
import numpy as np
import xarray as xr

def plsr():
    model = CP_PLSR(4)

    serology = MGH4D()["Serology"].stack(Sample = ("Subject", "Time"))
    X = np.transpose(serology.values)

    functional = MGH4D()["Functional"].stack(Sample = ("Subject", "Time"))
    Y = np.transpose(functional.values)

    model.fit(X, Y)

    pass


