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

    # read nc file:
    data = open_dataset(fpath, engine=config.nc_engine, **kwargs)
    if preprocess is not None:
        data = preprocess(data)

    dvrs = {}
    n_times = {}
    for v, c in var2ncvar.items():
        if c not in data:
            continue
        elif v == FV.U or v == FV.V:
            assert var2ncvar[FV.U] in data and var2ncvar[FV.V] in data, (
                f"Both {c} and {var2ncvar[FV.V]} must be in data."
            )
            if v == FV.U:
                assert var2ncvar[FV.WS] not in data, (
                    f"{var2ncvar[FV.WS]} must not be in data when {c} and {var2ncvar[FV.V]} are."
                )
                assert var2ncvar[FV.WD] not in data, (
                    f"{var2ncvar[FV.WD]} must not be in data when {c} and {var2ncvar[FV.V]} are."
                )
                dms = data[c].dims
                assert dms == data[var2ncvar[FV.V]].dims, (
                    f"Dimensions of {c} and {var2ncvar[FV.V]} do not match, got {dms} and {data[var2ncvar[FV.V]].dims}, respectively."
                )
                uv = np.stack(
                    [
                        data[c].values,
                        data[var2ncvar[FV.V]].values,
                    ],
                    axis=-1,
                )
                if coord in dms:
                    di = dms.index(coord)
                    n_times[FV.U] = np.sum(~np.isnan(uv[..., 0]), axis=di)
                    n_times[FV.V] = np.sum(~np.isnan(uv[..., 1]), axis=di)
                    n_times[FV.WS] = np.sum(~np.any(np.isnan(uv), axis=-1), axis=di)
                    dvrs[FV.U] = (dms, np.nansum(uv[..., 0], axis=di))
                    dvrs[FV.V] = (dms, np.nansum(uv[..., 1], axis=di))
                    dvrs[FV.WS] = (dms, np.nansum(np.linalg.norm(uv, axis=-1), axis=di))
                else:
                    n_times[FV.U] = None
                    n_times[FV.V] = None
                    n_times[FV.WS] = None
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
                    dvrs[FV.WS] = (dms, np.linalg.norm(uv, axis=-1))
        elif v == FV.WS or v == FV.WD:
            assert var2ncvar[FV.WD] in data and var2ncvar[FV.WS] in data, (
                f"Both {c} and {var2ncvar[FV.WD]} must be in data."
            )
            if v == FV.WS:
                dms = data[c].dims
                assert dms == data[var2ncvar[FV.WD]].dims, (
                    f"Dimensions of {c} and {var2ncvar[FV.WD]} do not match, got {dms} and {data[var2ncvar[FV.WD]].dims}, respectively."
                )
                ws = data[c].values
                uv = wd2uv(data[var2ncvar[FV.WD]].values, ws)
                if coord in dms:
                    di = dms.index(coord)
                    n_times[FV.WS] = np.sum(~np.isnan(ws), axis=di)
                    n_times[FV.U] = np.sum(~np.any(np.isnan(uv), axis=-1), axis=di)
                    n_times[FV.V] = n_times[FV.WS]
                    dvrs[FV.WS] = (dms, np.nansum(ws, axis=di))
                    dvrs[FV.U] = (dms, np.nansum(uv[..., 0], axis=di))
                    dvrs[FV.V] = (dms, np.nansum(uv[..., 1], axis=di))
                else:
                    n_times[FV.WS] = None
                    n_times[FV.U] = None
                    n_times[FV.V] = None
                    dvrs[FV.WS] = (dms, ws)
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
        else:
            d = data[c].values
            dms = data[c].dims
            if coord in dms:
                di = dms.index(coord)
                n_times[v] = np.sum(~np.isnan(d), axis=di)
                dvrs[v] = (dms, np.nansum(d, axis=di))
            else:
                n_times[v] = None
                dvrs[v] = (dms, d)

    crds = {}
    for dms, __ in dvrs.values():
        for d in dms:
            if d != coord and d not in crds and d in data.coords:
                crds[d] = data[d].values

    return crds, dvrs, n_times


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
    n_times = {}
    for hcrds, hdvrs, hn_times in map_with_engine(
        _read_nc,
        files,
        coord=coord,
        var2ncvar=v2nc,
        preprocess=preprocess,
        **kwargs,
    ):
        for v, t in hn_times.items():
            if v not in n_times:
                n_times[v] = t
            elif t is not None:
                assert n_times[v] is not None, (
                    f"Inconsistent n_times for variable {v}, got None and {t}"
                )
                assert t.shape == n_times[v].shape, (
                    f"Inconsistent n_times shape for variable {v}, got {n_times[v].shape} and {t.shape}"
                )
                n_times[v] += t
            else:
                n_times[v] = None
        for c, d in hcrds.items():
            if c not in crds:
                crds[c] = d
            elif not np.all(crds[c] == d):
                raise ValueError(f"Coordinate {c} does not match between files.")
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
            dvrs[v][1] /= n_times[v]

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
