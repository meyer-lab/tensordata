def test_atyeo():
    from ..atyeo import data
    
    dx = data()
    assert len(dx.sel(Antigen = 'S').shape) == 2


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

def test_kaplonek():
    from ..kaplonek import SpaceX4D, MGH4D
    
    sx = SpaceX4D()
    assert len(sx.sel(Receptor='IgA').shape) == 3
    
    mx = MGH4D()
    assert len(mx.sel(Antigen='CMV').shape) == 3

def test_kaplonekVaccine():
    from ..kaplonekVaccine import data

    dx = data()
    assert len(dx.sel(Antigen='RBD').shape) == 2
    assert len(dx.sel(Receptor='IgG3').shape) == 2

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