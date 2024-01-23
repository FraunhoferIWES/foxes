def write_nc(ds, fpath, round="auto", complevel=5, verbosity=1, **kwargs):
    """
    Writes a dataset to netCFD file

    Parameters
    ----------
    fpath: str
        Path to the output file, should be nc
    round: dict or str, optional
        The rounding definitions, or auto for
        default settings
    complevel: int
        The compression level
    verbosity: int
        The verbosity level, 0 = silent
    kwargs: dict, optional
        Additional parameters for xarray.to_netcdf

    """

    if round is not None:
        for v in ds.coords.keys():
            if v in round:
                if verbosity > 1:
                    print(f"Rounding {v} to {round[v]} decimals")
                ds[v].data = ds[v].data.round(decimals=round[v])
        for v in ds.data_vars.keys():
            if v in round:
                if verbosity > 1:
                    print(f"Rounding {v} to {round[v]} decimals")
                ds[v].data = ds[v].data.round(decimals=round[v])

    if verbosity > 0:
        print("Writing file", fpath)

    enc = {k: {"zlib": True, "complevel": complevel} for k in ds.data_vars}
    ds.to_netcdf(fpath, encoding=enc, **kwargs)
