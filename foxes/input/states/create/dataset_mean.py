import argparse
import numpy as np
from xarray import open_dataset, Dataset
from pathlib import Path

from foxes.core import map_with_engine, Engine
from foxes.config import config
from foxes.utils import wd2uv, write_nc
import foxes.variables as FV


def _read_nc(
    fpath,
    coord,
    var2ncvar,
    preprocess=None,
    **kwargs,
):
    """Help function to read netCDF files with xarray."""

    data = open_dataset(fpath, engine=config.nc_engine, **kwargs)
    if preprocess is not None:
        data = preprocess(data)

    cU, cV = var2ncvar[FV.U], var2ncvar[FV.V]
    cWS, cWD = var2ncvar[FV.WS], var2ncvar[FV.WD]
    if cU in data or cV in data:
        assert cU in data and cV in data, (
            f"Both {cU} and {cV} must be in data if one is."
        )
        assert data[cU].dims == data[cV].dims, (
            f"Dimensions of {cU} and {cV} do not match, got {data[cU].dims} and {data[cV].dims}, respectively."
        )
        uv = np.stack([data[cU].values, data[cV].values], axis=-1)
        ws = np.linalg.norm(uv, axis=-1)
        dms = data[cU].dims
    elif cWS in data or cWD in data:
        assert cWS in data and cWD in data, (
            f"Both {cWS} and {cWD} must be in data if one is."
        )
        assert data[cWS].dims == data[cWD].dims, (
            f"Dimensions of {cWS} and {cWD} do not match, got {data[cWS].dims} and {data[cWD].dims}, respectively."
        )
        ws = data[cWS].values
        uv = wd2uv(data[cWD].values, ws)
        dms = data[cWS].dims
    else:
        raise ValueError(f"None of {cU}, {cV}, {cWS}, {cWD} found in data.")

    assert coord in data.sizes, (
        f"Coordinate {coord} not found in data dimensions {data.sizes}"
    )
    n = data.sizes[coord]
    if coord in dms:
        di = dms.index(coord)
        uv = np.sum(uv, axis=di)
        ws = np.sum(ws, axis=di)

    dvrs = {}
    dvrs[FV.U] = (dms, uv[..., 0])
    dvrs[FV.V] = (dms, uv[..., 1])
    dvrs[FV.WS] = (dms, ws)

    for v, c in var2ncvar.items():
        if v not in (FV.U, FV.V, FV.WS):
            d = data[c].values
            dms = data[c].dims
            if coord in dms:
                di = dms.index(coord)
                d = np.sum(d, axis=di)
            dvrs[v] = (dms, d)

    crds = {}
    for dms, __ in dvrs.values():
        for d in dms:
            if d not in crds and d in data.coords:
                crds[d] = data[d].values

    return crds, dvrs, n


def create_dataset_mean(
    data_source,
    coord,
    var2ncvar,
    to_file=None,
    preprocess=None,
    verbosity=1,
    **kwargs,
):
    """
    Create dataset mean state data and optionally write to file.

    Parameters
    ----------
    data_source: str or xarray.Dataset
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    coord: str
        Name of the coordinate which should be averaged over
    var2ncvar: dict
        Mapping from variable names to netCDF variable names. Will
        be searched for FV.WS, FV.WD, FV.U, FV.V
    to_file: str, optional
        If given, write the mean state to this file
    preprocess: callable, optional
        Preprocessing function with signature ds = preprocess(ds)
        which is applied to each dataset after reading.
    verbosity: int
        The verbosity level, 0 = silent
    kwargs: dict, optional
        Additional keyword arguments passed to xarray.open_dataset

    Returns
    -------
    data: xarray.Dataset
        The created mean state data

    """
    # extend names by defaults:
    v2nc = {v: v for v in {FV.WS, FV.WD, FV.U, FV.V, FV.TI, FV.RHO}}
    v2nc.update(var2ncvar)

    # find files:
    fpath = Path(data_source)
    if verbosity > 0:
        print(
            f"Creating dataset mean over dimension {coord} from files matching {fpath}"
        )
    prt = fpath.resolve().parent
    glb = fpath.name
    while "*" in str(prt):
        glb = prt.name + "/" + glb
        prt = prt.parent
    files = sorted(list(prt.glob(glb)))

    # read files in parallel and compute mean:
    crds = {}
    dvrs = {}
    n = 0
    for hcrds, hdvrs, hn in map_with_engine(
        _read_nc,
        files,
        coord=coord,
        var2ncvar=v2nc,
        preprocess=preprocess,
        **kwargs,
    ):
        n += hn
        for c, d in hcrds.items():
            if c not in crds and c != coord:
                crds[c] = d
        for v, (dms, d) in hdvrs.items():
            if v not in dvrs or coord not in dvrs[v][0]:
                dvrs[v] = [dms, d]
            elif dms != dvrs[v][0]:
                raise ValueError(
                    f"Dimensions for variable {v} do not match between files, got {dms} and {dvrs[v][0]}"
                )
            else:
                dvrs[v][1] += d
    for v in dvrs:
        if coord in dvrs[v][0]:
            dvrs[v][0] = tuple(d for d in dvrs[v][0] if d != coord)
            dvrs[v][1] /= n

    data = Dataset(
        coords=crds,
        data_vars={v: (dms, d) for v, (dms, d) in dvrs.items()},
    )

    if to_file is not None:
        write_nc(data, to_file, nc_engine=config.nc_engine, verbosity=verbosity)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nc_files",
        help="NetCDF file pattern",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--coord",
        help="The coordinate to average over",
        type=str,
        default="time",
    )
    parser.add_argument(
        "-v",
        "--var2ncvar",
        help="Variable to netCDF variable name mapping, format: var1:ncvar1,var2:ncvar2,...",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--to_file",
        help="If given, write the mean state to this file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="The engine",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--n_cpus",
        help="The number of cpus",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    v2nc = {}
    for vn in args.var2ncvar:
        v, nc = vn.split(":")
        v2nc[v] = nc

    def _run():
        return create_dataset_mean(
            data_source=args.nc_files,
            coord=args.coord,
            var2ncvar=v2nc,
            to_file=args.to_file,
            preprocess=None,
            verbosity=1,
        )

    if args.engine is not None:
        with Engine.new(args.engine, n_procs=args.n_cpus):
            data = _run()
    else:
        data = _run()

    print(data)


if __name__ == "__main__":
    main()
