import numpy as np
from xarray import Dataset
from pathlib import Path

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
    fpath = Path(fpath)
    if round is not None:
        crds = {}
        for v in ds.coords.keys():
            d = round.get(v, get_default_digits(v))
            if d is not None:
                if verbosity > 1:
                    print(f"File {fpath.name}: Rounding {v} to {d} decimals")
                crds[v] = np.round(ds[v].to_numpy(), d)
            else:
                crds[v] = ds[v].to_numpy()
        dvrs = {}
        for v in ds.data_vars.keys():
            d = round.get(v, get_default_digits(v))
            if d is not None:
                if verbosity > 1:
                    print(f"File {fpath.name}: Rounding {v} to {d} decimals")
                dvrs[v] = (ds[v].dims, np.round(ds[v].to_numpy(), d))
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
