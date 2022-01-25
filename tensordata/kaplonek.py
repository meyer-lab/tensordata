from os.path import join, dirname
import numpy as np
import pandas as pd

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
    MGH_subjects = data.values[:, 0]
    MGH_data = data.values[:, 1:]

    ant_rec_names = load_file("MGH_Features")
    ant_names = ant_rec_names.values[:, 0]
    MGH_rec_names = ant_rec_names.values[:, 1]

    _, unique_rec_ind = np.unique(MGH_rec_names, return_index=True)
    MGH_unique_rec_names = MGH_rec_names[sorted(unique_rec_ind)]

    _, unique_ant_ind = np.unique(ant_names, return_index=True)
    MGH_unique_ant_names = ant_names[sorted(unique_ant_ind)]

    return MGH_data, MGH_subjects, MGH_rec_names, MGH_unique_rec_names, MGH_unique_ant_names


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
