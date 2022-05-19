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

    dxs = data(xarray=True)
    assert len(dxs[0].sel(Receptor='IgG3').shape) == 2
    assert len(dxs[1].sel(Glycan='G2F').shape) == 1

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
    from ..kaplonek import SpaceX, MGH
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
    mx = MGH(xarray=True)
    assert len(mx.sel(Antigen='CMV').shape) == 2
