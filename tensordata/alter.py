""" Data import and processing. """
from functools import reduce
from functools import lru_cache
from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd
from .__init__ import Bunch


path_here = dirname(dirname(__file__))


def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/alter2018/" + name + ".csv"), delimiter=",", comment="#")

    return data


def importLuminex(antigen=None):
    """ Import the Luminex measurements. Subset if only a specific antigen is needed. """
    df = load_file("data-luminex")
    df = pd.melt(df, id_vars=["subject"])

    if antigen is not None:
        df = df[df["variable"].str.contains(antigen)]
        df["variable"] = df["variable"].str.replace("." + antigen, "", regex=True)

        # Filter out bad antigen matches
        df = df[~df["variable"].str.contains("235")]
        df = df[~df["variable"].str.contains("244")]
        df = df[~df["variable"].str.contains("Kif")]
        df = df[~df["variable"].str.contains("delta3711")]

    return df


def importGlycan():
    """ Import the glycan measurements. """
    df = load_file("data-glycan-gp120")
    dfAxis = load_file("meta-glycans")
    df = pd.melt(df, id_vars=["subject"])

    glycan = dfAxis["glycan"].to_list()

    return glycan, df


def importIGG():
    """ Import the IgG measurements. """
    df = load_file("data-luminex-igg")
    df = pd.melt(df, id_vars=["subject"])

    df["variable"] = df["variable"].str.replace("IgG.", "", regex=True)

    return df


def getAxes():
    """ Get each of the axes over which the data is measured. """
    subjects = load_file("meta-subjects")
    detections = load_file("meta-detections")
    antigen = load_file("meta-antigens")

    subjects = subjects["subject"].to_list()
    detections = detections["detection"].to_list()
    antigen = antigen["antigen"].to_list()

    return subjects, detections, antigen


functions = ["ADCD", "ADCC", "ADNP", "CD107a", "IFNγ", "MIP1β"]


def importFunction():
    """ Import functional data. """
    subjects, _, _ = getAxes()
    df = load_file("data-function")
    df.columns = ["subject"] + functions
    df_a = pd.DataFrame({"subject": subjects})

    df = df_a.merge(df, on="subject", how="left")

    idnum = [0, 1, 2, 3, 4, 5]
    mapped = dict(zip(functions, idnum))

    return df, mapped


def importAlterDF(function=True, subjects=False):
    """ Recreate Alter DF, Import Luminex, Luminex-IGG, Subject group pairs, and Glycan into DF"""
    df = importLuminex()
    lum = df.pivot(index="subject", columns="variable", values="value")
    _, df2 = importGlycan()
    glyc = df2.pivot(index="subject", columns="variable", values="value")

    # Should we import functions or classes?
    if function is True:
        func, _ = importFunction()
    else:
        func = None
    if subjects is True:
        func = load_file("meta-subjects")

    igg = importIGG()
    igg = igg.pivot(index="subject", columns="variable", values="value")
    subj = load_file("meta-subjects")["subject"]
    data_frames = [lum, glyc, igg, func, subj]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=["subject"], how="inner"), data_frames,)

    return df_merged


def selectAlter(Y, Y_pred, evaluation, subset=None):
    """ Subset Y for sets of subjects. """
    df = importAlterDF().dropna()
    subjects = getAxes()[0]

    idx = np.zeros(181, dtype=np.bool)
    for subject in df["subject"]:
        idx[subjects.index(subject)] = 1

    if subset is not None:
        idx = idx[subset]

    if evaluation == "Alter":
        Y, Y_pred = Y[idx], Y_pred[idx]
    elif evaluation == "notAlter":
        Y, Y_pred = Y[~idx], Y_pred[~idx]
    elif evaluation != "all":
        raise ValueError("Bad evaluation selection.")

    assert Y.shape == Y_pred.shape
    return Y, Y_pred


@lru_cache()
def createCube():
    """ Import the data and assemble the antigen cube. """
    subjects, detections, antigen = getAxes()
    cube = np.full([len(subjects), len(detections), len(antigen)], np.nan)

    IGG = importIGG()
    glycan, dfGlycan = importGlycan()
    glyCube = np.full([len(subjects), len(glycan)], np.nan)

    for k, curAnti in enumerate(antigen):
        lumx = importLuminex(curAnti)

        for _, row in lumx.iterrows():
            i = subjects.index(row["subject"])
            j = detections.index(row["variable"])
            cube[i, j, k] = row["value"]

    for _, row in dfGlycan.iterrows():
        i = subjects.index(row["subject"])
        j = glycan.index(row["variable"])
        glyCube[i, j] = row["value"]

    # Add IgG data on the end as another detection
    for _, row in IGG.iterrows():
        i = subjects.index(row["subject"])
        k = antigen.index(row["variable"])
        cube[i, -1, k] = row["value"]

    # Clip to 0 as there are a few strongly negative outliers
    # IIa.H/R were offset to negative, so correct that
    cube[:, 1:11, :] = np.clip(cube[:, 1:11, :], 0, 175000)

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    return cube, glyCube

def data(xarray = False):
    cube, glyCube = createCube()
    subjects, detections, antigen = getAxes()
    glycan, _ = importGlycan()

    if xarray:
        return xr.Dataset(
            {
                "Fc": (["Sample", "Receptor", "Antigen"], cube),
                "gp120": (["Sample", "Glycan"], glyCube),
            },
            coords = {"Sample": subjects, "Receptor": detections, "Antigen": antigen, "Glycan": glycan},
        )

    return Bunch(
        tensor=cube,
        matrix=glyCube,
        mode=["Sample", "Receptor", "Antigen", "Glycan"],
        axes=[subjects, detections, antigen, glycan],
    )