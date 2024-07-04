from foxes.variables import get_default_digits

def write_nc(ds, fpath, round={}, complevel=9, verbosity=1, **kwargs):
    """
    Writes a dataset to netCFD file

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
    if round is not None:
        for v in ds.coords.keys():
            d = round.get(v, get_default_digits(v))
            if d is not None:
                if verbosity > 1:
                    print(f"Rounding {v} to {d} decimals")
                ds[v].data = ds[v].data.round(decimals=d)
        for v in ds.data_vars.keys():
            d = round.get(v, get_default_digits(v))
            if d is not None:
                if verbosity > 1:
                    print(f"Rounding {v} to {d} decimals")
                ds[v].data = ds[v].data.round(decimals=d)

    if verbosity > 0:
        print("Writing file", fpath)

    enc = {k: {"zlib": True, "complevel": complevel} for k in ds.data_vars}
    ds.to_netcdf(fpath, encoding=enc, **kwargs)
