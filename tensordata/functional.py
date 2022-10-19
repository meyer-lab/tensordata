from .kaplonek import MGH4D, SpaceX4D
from .zohar import data as Zohar
from tensorly.regression.cp_plsr import *
import numpy as np
import xarray as xr

def plsr():
    model = CP_PLSR(4)

    serology = MGH4D()["Serology"].stack(Sample = ("Subject", "Time"))
    finite_ser_ind = np.all(np.isfinite(serology.values), axis=(0,1))
    

    functional = MGH4D()["Functional"].stack(Sample = ("Subject", "Time"))
    finite_func_ind = np.all(np.isfinite(functional.values), axis=0)
    
    finite_ind = np.logical_and(finite_ser_ind, finite_func_ind)
    

    finite_func = functional.isel(Sample=finite_ind)
    finite_ser = serology.isel(Sample=finite_ind)
    X = np.transpose(finite_ser.values)
    Y = np.transpose(finite_func.values)

    model.fit(X, Y)

    pass


