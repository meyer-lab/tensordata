import numpy as np
import xarray as xr

from .kaplonek import MGH4D, SpaceX4D
from .zohar import data as Zohar


def checkMissingess(cube):
    return 1 - np.sum(np.isfinite(cube)) / np.product(cube.shape)


def normalizeSubj(cube):
    cube -= np.nanmean(cube, axis=0)
    cube = cube / np.nanstd(cube, axis=0)
    return cube


# Specific to Kaplonek MGH
M_dict = {
    "Antigen": [
        "SARS.CoV2_N",
        "CoV.OC43",
        "Flu_HA",
        "SARS.CoV2_S1",
        "Ebola_gp",
        "CMV",
        "SARS.CoV2_S",
        "SARS.CoV2_S2",
        "SARS.CoV2_RBD",
    ]
}

# Specific to Kaplonek SpaceX
S_dict = {
    "Antigen": [
        "CoV.HKU1_S",
        "CoV.OC43_RBD",
        "CoV.HKU1_RBD",
        "CoV.OC43_S",
        "SARS.CoV2_S",
        "SARS.CoV2_S1",
        "SARS.CoV2_RBD",
        "SARS_RBD",
        "SARS.CoV2_S2",
        "Flu_HA",
        "Ebola_gp",
        "MERS_RBD",
        "SARS_S",
        "MERS_S",
    ]
}

# Specific to Zohar
Z_dict = {
    "Antigen": [
        "SARS.CoV2_S",
        "SARS.CoV2_RBD",
        "SARS.CoV2_N",
        "SARS.CoV2_S1",
        "SARS.CoV2_S2",
        "SARS.CoV2_S1trimer",
    ]
}

""" Set consistent naming between the three DataArrays """


def serology_rename():
    M, S, Z = MGH4D()["Serology"], SpaceX4D(), Zohar()

    M = normalizeSubj(M)
    S = normalizeSubj(S)
    Z = normalizeSubj(Z)

    return M.assign_coords(M_dict), S.assign_coords(S_dict), Z.assign_coords(Z_dict)


""" Assemble the concatenated COVID tensor in 3D """


def importConcat():
    M, S, Z = serology_rename()
    cube = S.combine_first(M).combine_first(Z)
    cube = normalizeSubj(cube)

    return cube, M, S, Z


def sharedElements(occurence: int, *args):
    """Return entries that appears occurence time among the lists in args"""
    from collections import Counter

    cnt = Counter()
    for arr in args:
        if isinstance(arr, xr.DataArray):
            arr = arr.to_numpy()
        cnt.update(arr)
    return [k for k in cnt if cnt[k] >= occurence]


def concat4D():
    M = MGH4D()["Serology"].assign_coords(M_dict)
    # M = normalizeSubj(M)
    M = M.rename({"Subject": "Subject_MGH", "Time": "Time_MGH"})
    M.name = "MGH"

    S = SpaceX4D().assign_coords(S_dict)
    S = S.rename({"Subject": "Subject_SpaceX", "Time": "Time_SpaceX"})
    S.name = "SpaceX"

    Z = Zohar().assign_coords(Z_dict)
    Z = Z.rename({"Sample": "Sample_Zohar"})
    Z.name = "Zohar"

    common_receptors = sharedElements(2, M["Receptor"], S["Receptor"], Z["Receptor"])
    common_antigens = sharedElements(2, M["Antigen"], S["Antigen"], Z["Antigen"])

    def commonRA(A):
        A = A.where(A.Receptor.isin(common_receptors), drop=True)
        A = A.where(A.Antigen.isin(common_antigens), drop=True)
        return A

    return xr.combine_by_coords([commonRA(M), commonRA(S), commonRA(Z)])
