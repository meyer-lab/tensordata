import numpy as np

def test_atyeo():
    from ..atyeo import data
    
    dx = data()
    assert len(dx.sel(Antigen = 'S').shape) == 2
    assert np.isclose(dx.loc["10", "RBD", "IgG2"], 2331)
    assert np.isclose(dx.loc["3", "S", "FcRg2A"], 340324.5)


def test_alter():
    from ..alter import data
    
    ds = data()
    assert "Fc" in ds
    assert "gp120" in ds

    dx = ds.to_array()
    assert len(dx.sel(variable='Fc').shape) == 4
    assert len(dx.sel(variable='gp120').shape) == 4

def test_zohar():
    from ..zohar import data 

    dx = data()
    assert len(dx.sel(Antigen='S1').shape) == 2
    assert len(dx.sel(Receptor='IgG3').shape) == 2
    assert np.isclose(dx.loc["4_36", "RBD", "IgA2"], 31434.25)
    assert np.isclose(dx.loc["1_77", "S2", "IgG2"], 69874.5)

def test_kaplonek():
    from ..kaplonek import SpaceX4D, MGH4D
  
    sx = SpaceX4D()
    assert len(sx.sel(Receptor='IgA').shape) == 3
    assert np.isclose(sx.loc[123947, "CoV2_S", "IgM", 0], 3.524688150)
    assert np.isclose(sx.loc[124273, "MERS_S", "IgM", 0], 5.002234117)
    
    mx = MGH4D()
    assert len(mx.sel(Antigen='CMV').shape) == 3
    assert np.isclose(mx.loc["C4-014", "CMV", "IgG1", "D3"], 2.88166991)
    assert np.isclose(mx.loc["C4-342", "SARS.CoV2_N", "IgG2", "D0"], 4.11416750)

def test_kaplonekVaccine():
    from ..kaplonekVaccine import data

    dx = data()
    assert len(dx.sel(Antigen='RBD').shape) == 2
    assert len(dx.sel(Receptor='IgG3').shape) == 2
    assert np.isclose(dx.loc[26.12784657, "Gamma", "IgG1"], 1307)
    assert np.isclose(dx.loc[47.21805832, "S1", "IgG3"], 623.5)

def test_jones():
    from ..jones import process_RA_Tensor, make_RA_Tensor
    process_RA_Tensor()
    RA_xa = make_RA_Tensor()
    print(RA_xa.shape)
    assert len(RA_xa.shape) == 4

def test_serology():
    from ..serology import concat4D
    dat = concat4D()
    assert len(dat["MGH"].shape) == 4
    assert len(dat["SpaceX"].shape) == 4
    assert all(dat["MGH"]["Receptor"] == dat["Zohar"]["Receptor"])

def test_chung():
    from ..chung import data

    dx = data()
    assert len(dx.sel(Antigen='SARS2 Trimer').shape) == 2
    assert len(dx.sel(Receptor='IgM').shape) == 2
    assert np.isclose(dx.loc["KK121", "SARS2 S2", "Pan IgG"], 57568)
    assert np.isclose(dx.loc["CP04", "MERS NP", "IgM"], 142382)