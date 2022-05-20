from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd
from .__init__ import Bunch

path_here = dirname(dirname(__file__))

# import .read_csv
# organize data samplexantigenxreceptor
# find indices of each receptor and antigen, store


def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/kaplonek2021/" + name + ".csv"), delimiter=",", comment="#")

    return data


def importSpaceX():
    data = load_file("SpaceX_Sero.Data")
    SX_subjects = data.values[:, 0].astype(object)
    SX_data = data.values[:, 1:]

    ant_rec_names = load_file("SpaceX_Features")
    ant_names = ant_rec_names.values[:, 2]
    SX_rec_names = ant_rec_names.values[:, 3]

    _, unique_rec_ind = np.unique(SX_rec_names, return_index=True)
    SX_unique_rec_names = SX_rec_names[sorted(unique_rec_ind)]

    _, unique_ant_ind = np.unique(ant_names, return_index=True)
    SX_unique_ant_names = ant_names[sorted(unique_ant_ind)]

    return SX_data, SX_subjects, SX_rec_names, SX_unique_rec_names, SX_unique_ant_names


def cubeSpaceX():
    [SX_data, SX_subjects, SX_rec_names, SX_unique_rec_names, SX_unique_ant_names] = importSpaceX()

    rec_ind = np.zeros((SX_unique_rec_names.size, int(SX_rec_names.size / SX_unique_rec_names.size))).astype(int)

    for xx in range(SX_unique_rec_names.size):
        rec_index = np.where(SX_rec_names == SX_unique_rec_names[xx])
        rec_index = np.array(rec_index)
        rec_ind[xx, :] = rec_index

    SX_cube = np.zeros((SX_data[:, 0].size, SX_unique_rec_names.size, rec_ind[0, :].size))

    for subject_ind in range(np.size(SX_cube, 0)):
        for receptor_ind in range(np.size(SX_cube, 1)):
            SX_cube[subject_ind, receptor_ind, :] = SX_data[subject_ind, rec_ind[receptor_ind, :]]

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(SX_cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(SX_cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(SX_cube), axis=(1, 2)))

    assert(SX_cube[116, 5, 0] == SX_data[116, 70])
    assert(SX_cube[0, 0, 13] == SX_data[0, 13])
    assert(SX_cube[66, 3, 4] == SX_data[66, 46])

    return SX_cube

def SpaceX(xarray = False):
    [_, SX_subjects, _, SX_unique_rec_names, SX_unique_ant_names] = importSpaceX()
    if xarray:
        return xr.DataArray(cubeSpaceX(), dims=("Sample", "Receptor", "Antigen"),
                            coords={"Sample":SX_subjects, "Receptor":SX_unique_rec_names, "Antigen":SX_unique_ant_names})
    return Bunch(
        tensor = cubeSpaceX(),
        axes = [SX_subjects, SX_unique_rec_names, SX_unique_ant_names]
    )


def flattenSpaceX():

    _, SX_subjects, _, SX_unique_rec_names, SX_unique_ant_names = importSpaceX()

    SX_cube = cubeSpaceX()

    SX_subxant_names = np.empty((SX_subjects.size * SX_unique_ant_names.size), dtype=object)
    SX_flatCube = np.zeros((SX_unique_rec_names.size, (SX_subjects.size * SX_unique_ant_names.size)))

    first_ind = 0
    last_ind = SX_unique_ant_names.size
    for subject_ind in range(SX_subjects.size):
        elong_subjects = [str(SX_subjects[subject_ind])] * 14
        combined_subant = np.stack((elong_subjects, SX_unique_ant_names), axis=1)
        SX_subxant_names[first_ind:last_ind] = combined_subant[:, 0] + ' ' + combined_subant[:, 1]

        for receptor_ind in range(SX_unique_rec_names.size):
            SX_flatCube[receptor_ind, first_ind:last_ind] = SX_cube[subject_ind, receptor_ind, :]

        first_ind += SX_unique_ant_names.size
        last_ind += SX_unique_ant_names.size

    assert(SX_flatCube[5, 1624] == SX_cube[116, 5, 0])
    assert(SX_flatCube[0, 13] == SX_cube[0, 0, 13])
    assert(SX_flatCube[3, 928] == SX_cube[66, 3, 4])

    return SX_flatCube, SX_subxant_names, SX_unique_rec_names


def importMGH():
    data = load_file("MGH_Sero.Neut.WHO124.log10")
    MGH_subjects = data.values[:, 0].astype('str')
    MGH_data = data.values[:, 1:].astype('float64')

    ant_rec_names = load_file("MGH_Features")
    ant_names = ant_rec_names.values[:, 0].astype('str')
    MGH_rec_names = ant_rec_names.values[:, 1].astype('str')

    _, unique_rec_ind = np.unique(MGH_rec_names, return_index=True)
    MGH_unique_rec_names = MGH_rec_names[sorted(unique_rec_ind)]

    _, unique_ant_ind = np.unique(ant_names, return_index=True)
    MGH_unique_ant_names = ant_names[sorted(unique_ant_ind)]

    return MGH_data, MGH_subjects, MGH_rec_names, MGH_unique_rec_names, MGH_unique_ant_names


def MGHds():
    """ MGH import to a xr.Dataset, Will be the only one to keep for MGH """
    data = load_file("MGH_Sero.Neut.WHO124.log10")
    sample_ax = data.values[:, 0].astype('str')
    meas_ax = data.columns[1:-4]
    func_feats = data.columns[-4:]

    subj_ax = [s.split('_')[0] for s in sample_ax]
    day_ax = [s.split('_')[1] for s in sample_ax]

    meta = load_file("MGH_Features")
    Ag_ax = meta.values[:, 0].astype('str')
    rcp_ax = meta.values[:, 1].astype('str')

    fc = xr.DataArray(coords = [("Subject", np.unique(subj_ax)), ("Day", np.unique(day_ax)), ("Antigen", np.unique(Ag_ax)), ("Receptor", np.unique(rcp_ax))])
    for ii, samp in enumerate(sample_ax):
        for jj, meas in enumerate(meas_ax):
            fc.loc[samp.split('_')[0], samp.split('_')[1], Ag_ax[jj], rcp_ax[jj]] = data.loc[ii, meas]

    funcdat = xr.DataArray(coords = [("Subject", np.unique(subj_ax)), ("Day", np.unique(day_ax)), ("Feature", func_feats)])
    for ii, samp in enumerate(sample_ax):
        for ffeat in func_feats:
            funcdat.loc[samp.split('_')[0], samp.split('_')[1], ffeat] = data.loc[ii, ffeat]
    return xr.Dataset({"Fc": fc, "functional": funcdat})


def cubeMGH():
    [MGH_data, MGH_subjects, MGH_rec_names, MGH_unique_rec_names, MGH_unique_ant_names] = importMGH()

    if MGH_data[0, :].size != MGH_rec_names.size:
        function_data = MGH_data[:, MGH_rec_names.size:]
        MGH_data = MGH_data[:, :MGH_rec_names.size]

    rec_ind = np.zeros((MGH_unique_rec_names.size, int(MGH_rec_names.size / MGH_unique_rec_names.size))).astype(int)

    for xx in range(MGH_unique_rec_names.size):
        rec_index = np.where(MGH_rec_names == MGH_unique_rec_names[xx])
        rec_index = np.array(rec_index)
        rec_ind[xx, :] = rec_index

    MGH_cube = np.zeros((MGH_data[:, 0].size, MGH_unique_rec_names.size, rec_ind[0, :].size))

    for subject_ind in range(np.size(MGH_cube, 0)):
        for receptor_ind in range(np.size(MGH_cube, 1)):
            MGH_cube[subject_ind, receptor_ind, :] = MGH_data[subject_ind, rec_ind[receptor_ind, :]]

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(MGH_cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(MGH_cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(MGH_cube), axis=(1, 2)))

    # Check data order (MGH)
    assert(MGH_cube[60, 5, 7] == MGH_data[60, 52])
    assert(MGH_cube[0, 9, 8] == MGH_data[0, 89])
    assert(MGH_cube[578, 4, 0] == MGH_data[578, 36])

    return MGH_cube, function_data


def flattenMGH():

    _, MGH_subjects, _, MGH_unique_rec_names, MGH_unique_ant_names = importMGH()

    MGH_cube, _ = cubeMGH()

    MGH_subxant_names = np.empty((MGH_subjects.size * MGH_unique_ant_names.size), dtype=object)
    MGH_flatCube = np.zeros((MGH_unique_rec_names.size, (MGH_subjects.size * MGH_unique_ant_names.size)))

    first_ind = 0
    last_ind = MGH_unique_ant_names.size
    for subject_ind in range(MGH_subjects.size):
        elong_subjects = [str(MGH_subjects[subject_ind])] * 9
        combined_subant = np.stack((elong_subjects, MGH_unique_ant_names), axis=1)
        MGH_subxant_names[first_ind:last_ind] = combined_subant[:, 0] + ' ' + combined_subant[:, 1]

        for receptor_ind in range(MGH_unique_rec_names.size):
            MGH_flatCube[receptor_ind, first_ind:last_ind] = MGH_cube[subject_ind, receptor_ind, :]

        first_ind += MGH_unique_ant_names.size
        last_ind += MGH_unique_ant_names.size

    assert(MGH_flatCube[5, 547] == MGH_cube[60, 5, 7])
    assert(MGH_flatCube[9, 8] == MGH_cube[0, 9, 8])
    assert(MGH_flatCube[4, 5202] == MGH_cube[578, 4, 0])

    return MGH_flatCube, MGH_subxant_names, MGH_unique_rec_names

def MGH(xarray = False):
    MGH_cube, function_data = cubeMGH()
    _, MGH_subjects, _, MGH_unique_rec_names, MGH_unique_ant_names = importMGH()
    if xarray:
        dat = xr.DataArray(MGH_cube, dims=("Sample", "Receptor", "Antigen"),
                            coords={"Sample":MGH_subjects, "Receptor":MGH_unique_rec_names, "Antigen":MGH_unique_ant_names})
        dat.attrs["functions"] = function_data
        return dat
    return Bunch(
        tensor = MGH_cube,
        mode=["Sample", "Receptor", "Antigen"],
        axes = [MGH_subjects, MGH_unique_rec_names, MGH_unique_ant_names],
        functions = function_data
    )

def MGH4D(xarray = False):
    cube, function_data = cubeMGH()
    _, sampleax, _, recs, Ags = importMGH()

    subjax = [s.split('_')[0] for s in sampleax]
    dayax = [s.split('_')[1] for s in sampleax]
    subju = np.unique(subjax)
    dayu = np.unique(dayax)

    cube4d = np.full((len(subju), len(dayu), cube.shape[1], cube.shape[2]), np.nan)
    for ii in range(len(sampleax)):
        cube4d[np.where(subju == subjax[ii])[0][0], np.where(dayu == dayax[ii])[0][0], :, :] = cube[ii, :, :]

    return xr.DataArray(cube4d, dims=("Subject", "Day", "Receptor", "Antigen"),
                       coords={"Subject": subju, "Day": dayu, "Receptor": recs, "Antigen": Ags})