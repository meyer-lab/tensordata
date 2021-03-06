from os.path import join, dirname
import numpy as np
import xarray as xr
import pandas as pd
from .__init__ import Bunch

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "tensordata/atyeo2020/" + name + ".csv"), delimiter=",", comment="#")

    return data


def getAxes():
    """ Get each of the axes over which the data is measured. """
    df = load_file("atyeo_covid")
    df = df.filter(regex='SampleID|Ig|Fc|SNA|RCA', axis=1)

    axes = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    axes = axes.columns.str.split(" ", expand = True)

    subject = df['SampleID']
    subject = list(subject[0:22])

    antigen = []
    receptor = []

    for row in axes:
        if (row[0] not in antigen):
            antigen.append(row[0])
        if (row[1] not in receptor):
            receptor.append(row[1])

    return subject, receptor, antigen
    

def createCube():
    """ Import the data and assemble the antigen cube. """
    subject, receptor, antigen = getAxes()
    cube = np.full([len(subject), len(receptor), len(antigen)], np.nan)
    
    df = load_file("atyeo_covid")
    df = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    df = df[0:len(subject)]

    for i, row in df.iterrows():
        for j in range(len(receptor)):
            rec =  df.filter(regex=receptor[j])
            cube[i,j] = rec.iloc[i,:]
    
    # Check that there are no slices with completely missing data        
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    return cube 


def flattenCube():
    """ Get and flatten the antigen cube. """
    subject, receptor, antigen = getAxes()
    
    cube = createCube()
    
    flatCube = np.full((len(receptor), (len(subject) * len(antigen))), np.nan)
    
    first_ind = 0
    last_ind = len(antigen)
    for subject_ind in range(len(subject)):
        for receptor_ind in range(len(receptor)):
            flatCube[receptor_ind, first_ind:last_ind] = cube[subject_ind, receptor_ind,:]

        first_ind += len(antigen)
        last_ind += len(antigen)
        
    return flatCube


def data(xarray = False):
    subject, receptor, antigen = getAxes()
    if xarray:
        return xr.DataArray(createCube(), dims=("Sample", "Receptor", "Antigen"),
                            coords={"Sample":subject, "Receptor":receptor, "Antigen":antigen})
    return Bunch(
        tensor=createCube(),
        mode=["Sample", "Receptor", "Antigen"],
        axes=[subject, receptor, antigen],
    )