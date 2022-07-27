__version__ = '0.0.3'

class Bunch(dict):
    """ A Bunch, exposing dict keys as a keys() method.
    Definition from scikit-learn. """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def xr_to_bunch(data):
    import xarray as xr
    assert isinstance(data, xr.DataArray)
    return Bunch(
        tensor = data.to_numpy(),
        mode = list(data.coords.dims),
        axes = [data.coords[dim].values for dim in data.coords.dims],
    )