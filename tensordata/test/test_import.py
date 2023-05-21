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
    assert "Functional" in ds

    assert len(ds["Sample"]) == 181
    assert len(ds["Receptor"]) == 22
    assert len(ds["Glycan"]) == 25
    assert len(ds["Function"]) == 6

    assert np.isclose(ds["Fc"].loc[753551, "FcgRIIIa", "gp120.PVO"], 14.3)
    assert np.isclose(ds["Fc"].loc[795613, "FcgRIIIa.V158", "gp120.TRO"], 13146.1 + 681.9)
    assert np.isclose(ds["Fc"].loc[190469, "IgG", "gp120.MN"], 39235.80337)
    assert np.isclose(ds["gp120"].loc[184538, "G1FB"], 3.5)
    assert np.isclose(ds["gp120"].loc[916549, "G2.total"], 38.1)
    assert np.isclose(ds["Functional"].loc[513662, "ADNP"], 14.42)
    assert np.isclose(ds["Functional"].loc[472789, "CD107a"], 13.02)

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
    
    mx = MGH4D()['Serology']
    assert len(mx.sel(Antigen='CMV').shape) == 3
    assert np.isclose(mx.loc["C4-014", "CMV", "IgG1", "D3"], 2.88166991)
    assert np.isclose(mx.loc["C4-342", "SARS.CoV2_N", "IgG2", "D0"], 4.11416750)

    mx_func = MGH4D()['Functional']
    assert np.isclose(mx_func.loc["C4-066", "ADCD", "D0"], 28856.447735)
    assert np.isclose(mx_func.loc["C4-372", "ADNP", "D3"], 83251.3686252)

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
    assert np.isclose(dx.loc["KK121", "SARS2 S2", "PanIgG"], 57568)
    assert np.isclose(dx.loc["CP04", "MERS NP", "IgM"], 142382)