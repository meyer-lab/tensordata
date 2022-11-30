from .kaplonek import MGH4D, SpaceX4D
from .zohar import data as Zohar
from tensorly.regression.cp_plsr import *
import numpy as np
import xarray as xr
from tensorly.decomposition._cp import CPTensor, cp_to_tensor
import pandas as pd

def calcR2X(tOrig, tFac):
    tMask = np.isfinite(tOrig)
    tIn = np.nan_to_num(tOrig)
    if isinstance(tFac, CPTensor):
        tFac = cp_to_tensor(tFac)
    vTop = np.linalg.norm(tFac * tMask - tIn) ** 2.0
    vBottom = np.linalg.norm(tIn) ** 2.0
    return 1.0 - vTop / vBottom



def plsr(R):
    serology = MGH4D()["Serology"].stack(Sample = ("Subject", "Time"))
    finite_ser_ind = np.all(np.isfinite(serology.values), axis=(0,1))

    functional = MGH4D()["Functional"].stack(Sample = ("Subject", "Time"))
    finite_func_ind = np.all(np.isfinite(functional.values), axis=0)
    
    finite_ind = np.logical_and(finite_ser_ind, finite_func_ind)
    
    finite_func = functional.isel(Sample=finite_ind)
    finite_ser = serology.isel(Sample=finite_ind)
    X = np.transpose(finite_ser.values)
    Y = np.transpose(finite_func.values)

    rr = R + 1
    R2X_arr = np.zeros(R)
    R2Y_arr = np.zeros(R)

    for rr in range(1, rr):
        model = CP_PLSR(rr)
        model.fit(X, Y)
        Xrecon = CPTensor((None, model.X_factors))
        Yrecon = CPTensor((None, model.Y_factors))
        XX = X - np.mean(X, axis=0)
        YY = Y - np.mean(Y, axis=0)
        R2X = calcR2X(XX, Xrecon)
        R2Y = calcR2X(YY, Yrecon)
        R2X_arr[rr-1] = R2X
        R2Y_arr[rr-1] = R2Y

    return R2X_arr, R2Y_arr


