from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xr
from .__init__ import Bunch

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/chung2021/" + name + ".csv"), delimiter=",", comment="#")

    return data


def importData():
    data = load_file("fig6")
    subjects = data.values[:, 0].astype('str')
    values = data.values[:, 5:].astype('float64')

    names = load_file("fig1")
    rec_names = names.columns.values[12:].astype('str')
    length = len(rec_names)
    unique_rec_names = rec_names[0:length:16]
    
    ant_names = names.values[0,12:].astype('str')
    for ii, str in enumerate(ant_names):
        ant_names[ii] = str.replace("\r\n", "")
    unique_ant_names = ant_names[:16]

    return  values, subjects, rec_names, unique_rec_names, ant_names, unique_ant_names


def makeCube():
    data, _, rec_names, unique_rec_names, _, unique_ant_names = importData()

    rec_ind = np.zeros((unique_rec_names.size, int(rec_names.size / unique_rec_names.size))).astype(int)

    for ii in range(unique_rec_names.size):
        rec_index = np.where(rec_names == unique_rec_names[ii])
        rec_index = np.array(rec_index)
        for jj in range(unique_ant_names.size):
            rec_ind[ii, jj] = rec_index + jj

    cube = np.zeros((data[:, 0].size, unique_rec_names.size, rec_ind[0, :].size))

    for subject_ind in range(np.size(cube, 0)):
        for receptor_ind in range(np.size(cube, 1)):
            cube[subject_ind, receptor_ind, :] = data[subject_ind, rec_ind[receptor_ind, :]]

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    return cube

def data(xarray = False):
    cube = makeCube()
    _, subjects, _, unique_rec_names, _, unique_ant_names = importData()
    if xarray:
        dat = xr.DataArray(cube, dims=("Sample", "Receptor", "Antigen"),
                            coords={"Sample":subjects, "Receptor":unique_rec_names, "Antigen":unique_ant_names})
        return dat
    return Bunch(
        tensor = cube,
        mode=["Sample", "Receptor", "Antigen"],
        axes = [subjects, unique_rec_names, unique_ant_names],
    )



    


