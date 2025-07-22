import numpy as np
from xarray import Dataset
from pathlib import Path
from numpy._core._exceptions import _UFuncNoLoopError

from foxes.variables import get_default_digits


def write_nc(
    ds,
    fpath,
    round={},
    complevel=9,
    verbosity=1,
    **kwargs,
):
    """
    Writes a dataset to netCDF file

    Parameters
    ----------
    fpath: str
        Path to the output file, should be nc
    round: dict
        The rounding digits, falling back to defaults
        if variable not found
    complevel: int
        The compression level
    verbosity: int
        The verbosity level, 0 = silent
    kwargs: dict, optional
        Additional parameters for xarray.to_netcdf

    """

    def _round(x, v, d):
        """Helper function to round values"""
        if d is not None:
            if verbosity > 1:
                print(f"File {fpath.name}: Rounding {v} to {d} decimals")
            try:
                x = x.astype(np.float32)
            except ValueError:
                pass
            try:
                return np.round(x, d)
            except _UFuncNoLoopError:
                pass
        return x

    fpath = Path(fpath)
    if round is not None:
        crds = {}
        for v, x in ds.coords.items():
            d = round.get(v, get_default_digits(v))
            crds[v] = _round(x.to_numpy(), v, d)
        dvrs = {}
        for v, x in ds.data_vars.items():
            d = round.get(v, get_default_digits(v))
            dvrs[v] = (x.dims, _round(x.to_numpy(), v, d))
        ds = Dataset(coords=crds, data_vars=dvrs)

    enc = None
    if complevel is not None and complevel > 0:
        if verbosity > 1:
            print(f"File {fpath.name}: Compression level = {complevel}")
        enc = {k: {"zlib": True, "complevel": complevel} for k in ds.data_vars}

    if verbosity > 0:
        print("Writing file", fpath)

    from foxes.config import config

    kw = dict(encoding=enc, engine=config.nc_engine)
    kw.update(kwargs)
    ds.to_netcdf(fpath, **kw)
