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
    from ..zohar import data
    d = data()
    shp = d.tensor.shape
    for ii in range(4):
        assert shp[ii] == len(d.axes[ii])
