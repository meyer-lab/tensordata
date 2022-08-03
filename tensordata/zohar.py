"""Import Zohar data, tensor formation, plotting raw data."""
from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd
from .__init__ import Bunch

path_here = dirname(dirname(__file__))

def pbsSubtractOriginal():
    """ Paper Background subtract, will keep all rows for any confusing result. """
    Cov = pd.read_csv(join(path_here, "tensordata/zohar2020/ZoharCovData.csv"), index_col=0)
    # 23 (0-> 23) is the start of IgG1_S
    Demographics = Cov.iloc[:, 0:23]
    Serology = Cov.iloc[:, 23::]
    Serology -= Serology.loc["PBS"].values.squeeze()
    df = pd.concat([Demographics, Serology], axis=1)
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["week"] = np.array(df["days"] // 7 + 1.0, dtype=int)
    df["patient_ID"] = df["patient_ID"].astype('int32')
    df["group"] = pd.Categorical(df["group"], ["Negative", "Mild", "Moderate", "Severe", "Deceased"])
    df = df.sort_values(by=["group", "days", "patient_ID"])
    df = df.reset_index()
    # Get rid of any data that doesn't have a time component (i.e. "nan" for day)
    df = df.dropna(subset=["days"]).reset_index(drop=True)
    return df


def to_slice(subjects, df):
    Rlabels, AgLabels = dimensionLabel3D()
    tensor = np.full((len(subjects), len(AgLabels), len(Rlabels)), np.nan)
    missing = 0

    for rii, recp in enumerate(Rlabels):
        for aii, anti in enumerate(AgLabels):
            try:
                dfAR = df[recp + "_" + anti]
                dfAR = dfAR.groupby(by="patient").mean()
                dfAR = dfAR.reindex(subjects)
                tensor[:, aii, rii] = dfAR.values
            except KeyError:
                #print(recp + "_" + anti)
                missing += 1

    return tensor

def dayLabels():
    """ Returns day labels for 4D tensor"""
    df = pbsSubtractOriginal()
    days = np.unique(df["days"])
    return days

def Tensor4D():
    """ Create a 4D Tensor (Subject, Antigen, Receptor, Time) """
    df = pbsSubtractOriginal()
    subjects = np.unique(df['patient_ID'])
    Rlabels, AgLabels = dimensionLabel3D()
    days = np.unique(df["days"])
    ndf = df.iloc[:, np.hstack([[1,10], np.arange(23, len(df.columns))])]

    tensor = np.full((len(subjects), len(AgLabels), len(Rlabels), len(days)), np.nan) # 4D

    for i in range(len(ndf)):
        row = ndf.iloc[i, :]
        patient = np.where(row['patient_ID']==subjects)[0][0]
        day = np.where(row['days']==days)[0][0]
        for j in range(2, len(ndf.columns)):
            key = ndf.columns[j].split('_')
            try:
                rii = Rlabels.index(key[0])
                aii = AgLabels.index(key[1])
                tensor[patient, aii, rii, day] = ndf.iloc[i, j]
            except:
                pass

    tensor = np.clip(tensor, 10.0, None)
    tensor = np.log10(tensor)

    # Mean center each measurement
    tensor -= np.nanmean(tensor, axis=0)

    return tensor, np.array(df.index)

def Tensor3D():
    """ Create a 3D Tensor (Antigen, Receptor, Sample in time) """
    df = pbsSubtractOriginal()
    Rlabels, AgLabels = dimensionLabel3D()

    tensor = np.full((len(df), len(AgLabels), len(Rlabels)), np.nan)
    missing = 0

    for rii, recp in enumerate(Rlabels):
        for aii, anti in enumerate(AgLabels):
            try:
                dfAR = df[recp + "_" + anti]
                tensor[:, aii, rii] = dfAR.values
            except KeyError:
                missing += 1

    tensor = np.clip(tensor, 10.0, None)
    tensor = np.log10(tensor)

    # Mean center each measurement
    tensor -= np.nanmean(tensor, axis=0)

    return tensor, np.array(df.index)

def dimensionLabel3D():
    """Returns labels for receptor and antigens, included in the 4D tensor"""
    receptorLabel = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgA1",
        "IgA2",
        "IgM",
        "FcRalpha",
        "FcR2A",
        "FcR2B",
        "FcR3A",
        "FcR3B"
    ]
    antigenLabel = ["S", "RBD", "N", "S1", "S2", "S1 Trimer"]
    return receptorLabel, antigenLabel


def time_components_df(tfac, condition=None):
    subj = pbsSubtractOriginal()
    df = pd.DataFrame(tfac.factors[0])
    comp_names = ["Comp. " + str((i + 1)) for i in range(tfac.factors[0].shape[1])]
    df.columns = comp_names
    df['days'] = subj['days'].values
    df['group'] = subj['group'].values
    df['week'] = subj['week'].values
    if condition is not None:
        df = df.loc[(subj["group"] == condition).values, :]
    df = df.dropna()
    df = pd.melt(df, id_vars=['days', 'group', 'week'], value_vars=comp_names)
    df.rename(columns={'variable': 'Factors'}, inplace=True)
    return df


def data(xarray = False):
    df = pbsSubtractOriginal()
    subjects = np.unique(df['patient_ID'])
    tensor, _ = Tensor4D()
    receptorLabel, antigenLabel = dimensionLabel3D()
    days = dayLabels()

    if xarray:
        return xr.DataArray(tensor, dims=("Subject", "Antigen", "Receptor", "Days"),
                            coords={"Subject":subjects, "Antigen":antigenLabel, "Receptor":receptorLabel, "Days":days})

    return Bunch(
        tensor=tensor,
        mode=["Subject", "Antigen", "Receptor", "Days"],
        axes=[subjects, antigenLabel, receptorLabel, days],
    )

def data3D():
    tensor, samples = Tensor3D()
    receptorLabel, antigenLabel = dimensionLabel3D()

    return xr.DataArray(tensor, dims=("Sample", "Antigen", "Receptor"),
                        coords={"Sample":samples, "Antigen":antigenLabel, "Receptor":receptorLabel})

