def test_atyeo():
    from ..atyeo import data
    d = data()
    shp = d.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(d.axes[ii])

    dx = data(xarray = True)
    assert len(dx.sel(Antigen = 'S').shape) == 2


def test_alter():
    from ..alter import data
    d = data()
    shp = d.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(d.axes[ii])
    assert d.tensor.shape[0] == d.matrix.shape[0]
    assert d.matrix.shape[1] ==  len(d.axes[3])

    ds = data(xarray=True)
    assert "Fc" in ds
    assert "gp120" in ds

def test_zohar():
    from ..zohar import data, data3D
    d = data()
    shp = d.tensor.shape
    for ii in range(4):
        assert shp[ii] == len(d.axes[ii])

    dx = data(xarray=True)
    assert len(dx.sel(Antigen='S1').shape) == 3

    d3 = data3D()
    shp3 = d3.tensor.shape
    for ii in range(3):
        assert shp3[ii] == len(d3.axes[ii])

    dx3 = data3D(xarray=True)
    assert len(dx3.sel(Antigen='RBD').shape) == 2

def test_kaplonek():
    from ..kaplonek import SpaceX, MGH, MGHds
    s = SpaceX()
    shp = s.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(s.axes[ii])
    sx = SpaceX(xarray=True)
    assert len(sx.sel(Receptor='IgA').shape) == 2

    m = MGH()
    mhp = m.tensor.shape
    for ii in range(3):
        assert mhp[ii] == len(m.axes[ii])
    ds = MGHds()
    assert len(ds["Fc"].sel(Antigen='CMV').shape) == 3

def test_jones():
    from ..jones import process_RA_Tensor, make_RA_Tensor
    process_RA_Tensor()
    RA_xa = make_RA_Tensor()
    print(RA_xa.shape)
    assert len(RA_xa.shape) == 4
