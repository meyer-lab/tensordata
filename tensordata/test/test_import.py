def test_atyeo():
    from ..atyeo import data
    d = data()
    shp = d.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(d.axes[ii])

def test_alter():
    from ..alter import data
    d = data()
    shp = d.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(d.axes[ii])
    assert d.tensor.shape[0] == d.matrix.shape[0]
    assert d.matrix.shape[1] ==  len(d.axes[3])

def test_zohar():
    from ..zohar import data, data3D
    d = data()
    shp = d.tensor.shape
    for ii in range(4):
        assert shp[ii] == len(d.axes[ii])

    d3 = data3D()
    shp3 = d3.tensor.shape
    for ii in range(3):
        assert shp3[ii] == len(d3.axes[ii])

def test_kaplonek():
    from ..kaplonek import SpaceX, MGH
    s = SpaceX()
    shp = s.tensor.shape
    for ii in range(3):
        assert shp[ii] == len(s.axes[ii])
    m = MGH()
    mhp = m.tensor.shape
    for ii in range(3):
        assert mhp[ii] == len(m.axes[ii])