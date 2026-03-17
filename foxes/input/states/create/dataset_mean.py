import argparse
import numpy as np
from xarray import open_dataset, Dataset
from pathlib import Path

from foxes.core import get_engine, Engine
from foxes.config import config
from foxes.utils import uv2wd, wd2uv, write_nc
import foxes.variables as FV


def _read_nc(
    fpath,
    coord,
    var2ncvar,
    vname_mean_ws,
    vname_main_wd,
    wd_histo_minwidth=1.0,
    preprocess=None,
    **kwargs,
):
    """Help function to read netCDF files with xarray."""

    # read nc file:
    data = open_dataset(fpath, engine=config.nc_engine, **kwargs)
    if preprocess is not None:
        data = preprocess(data)

    dvrs = {}
    counts = {}
    wd_histo = None
    uv = None
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
                    a = list(range(len(uv.shape)))
                    a.remove(di)
                    uvsel = ~np.any(np.isnan(uv), axis=tuple(a))
                    s = [slice(None)] * len(uv.shape)
                    s[di] = uvsel
                    s = tuple(s)
                    a = tuple([m for ii, m in enumerate(uv.shape[:-1]) if ii != di])

                    counts[FV.U] = np.full(a, np.sum(uvsel), dtype=config.dtype_int)
                    counts[FV.V] = counts[FV.U]
                    counts[vname_mean_ws] = counts[FV.U]
                    dvrs[vname_mean_ws] = (
                        dms,
                        np.sum(np.linalg.norm(uv, axis=-1), axis=di),
                    )
                    uv = np.sum(uv[s], axis=di)
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
                else:
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
                    dvrs[vname_mean_ws] = (dms, np.linalg.norm(uv, axis=-1))
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
                    a = list(range(len(uv.shape)))
                    a.remove(di)
                    uvsel = ~np.any(np.isnan(uv), axis=tuple(a))
                    s = [slice(None)] * len(uv.shape)
                    s[di] = uvsel
                    s = tuple(s)
                    a = tuple([m for ii, m in enumerate(uv.shape[:-1]) if ii != di])

                    counts[vname_mean_ws] = np.sum(~np.isnan(ws), axis=di)
                    counts[FV.U] = np.full(a, np.sum(uvsel), dtype=config.dtype_int)
                    counts[FV.V] = counts[FV.U]
                    dvrs[vname_mean_ws] = (dms, np.nansum(ws, axis=di))
                    uv = np.sum(uv[s], axis=di)
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
                else:
                    dvrs[vname_mean_ws] = (dms, ws)
                    dvrs[FV.U] = (dms, uv[..., 0])
                    dvrs[FV.V] = (dms, uv[..., 1])
        else:
            d = data[c].values
            dms = data[c].dims
            if coord in dms:
                di = dms.index(coord)
                counts[v] = np.sum(~np.isnan(d), axis=di)
                dvrs[v] = (dms, np.nansum(d, axis=di))
            else:
                dvrs[v] = (dms, d)

    # compute wd histogram counts:
    if vname_main_wd is not None:
        wd = uv2wd(uv)
        wds = np.linspace(0.0, 360.0, 2 * int(180 / wd_histo_minwidth * 2))
        n_bins = len(wds) - 1
        wd_histo = np.zeros(wd.shape + (n_bins,), dtype=config.dtype_int)
        np.put_along_axis(
            wd_histo,
            np.searchsorted(wds, wd, side="right")[..., None] - 1,
            counts[FV.U].reshape(wd.shape + (1,)),
            axis=-1,
        )

    crds = {}
    for dms, __ in dvrs.values():
        for d in dms:
            if d != coord and d not in crds and d in data.coords:
                crds[d] = data[d].values

    return crds, dvrs, counts, wd_histo


def create_dataset_mean(
    data_source,
    coord,
    var2ncvar,
    vname_mean_ws=FV.MEAN_WS,
    vname_main_wd=FV.MAIN_WD,
    wd_histo_minwidth=1.0,
    wd_histo_maxwidth=30.0,
    add_uv=False,
    add_counts=False,
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
        Mapping from variable names to netCDF variable names
    vname_mean_ws: str
        The variable name to use for the mean wind speed
    vname_main_wd: str
        The variable name to use for the main wind direction
    wd_histo_minwidth: float
        The minimal wind direction histogramm bin width
    wd_histo_maxwidth: float
        The maximal wind direction histogramm bin width
    add_uv: bool
        Flag for adding U and V to the resulting data
    add_counts: bool
        Flag for adding the counts of each data variable
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

    :group: input.states.create

    """
    # prepare:
    engine = get_engine()
    crds = {}
    dvrs = {}
    counts = {}
    wd_histo = None

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

    # submit file reading and processing to workers:
    futures = [
        engine.submit(
            _read_nc,
            fpath,
            coord=coord,
            var2ncvar=v2nc,
            preprocess=preprocess,
            vname_mean_ws=vname_mean_ws,
            vname_main_wd=vname_main_wd,
            wd_histo_minwidth=wd_histo_minwidth,
            **kwargs,
        )
        for fpath in files
    ]

    def _eval_result(hcrds, hdvrs, hcounts, hwd_histo):
        """Helper function that evaluates single result"""
        nonlocal wd_histo, crds, dvrs, counts

        if hwd_histo is not None:
            if wd_histo is None:
                wd_histo = hwd_histo.copy()
            else:
                wd_histo += hwd_histo

        for v, t in hcounts.items():
            if v not in counts:
                counts[v] = t.copy() if t is not None else None
            elif t is not None:
                assert counts[v] is not None, (
                    f"Inconsistent counts for variable {v}, got None and {t}"
                )
                assert t.shape == counts[v].shape, (
                    f"Inconsistent counts shape for variable {v}, got {counts[v].shape} and {t.shape}"
                )
                counts[v] += t
            elif counts[v] is not None:
                raise ValueError(
                    f"Inconsistent counts for variable {v}, got {counts[v]} and None"
                )
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

    # Await and evaluate results:
    proc = -1
    if verbosity > 0:
        print(f"Reading and processing {len(files)} files...")
    for i in range(len(files)):
        _eval_result(*engine.await_result(futures.pop(0)))
        if verbosity > 0:
            hproc = int((i + 1) / len(files) * 100)
            if hproc > proc:
                proc = hproc
                print(f"Processed files: {proc}% ({i + 1}/{len(files)})")

    cnts = {}
    for v in dvrs:
        if coord in dvrs[v][0]:
            dvrs[v][0] = tuple(d for d in dvrs[v][0] if d != coord)
            dvrs[v][1] /= counts[v]
            cnts[f"counts_{v}"] = (dvrs[v][0], counts[v])

    uv = np.stack([dvrs[FV.U][1], dvrs[FV.V][1]], axis=-1)
    dvrs[FV.WS] = (dvrs[FV.U][0], np.linalg.norm(uv, axis=-1))
    dvrs[FV.WD] = (dvrs[FV.U][0], uv2wd(uv))
    cnts[f"counts_{FV.WS}"] = cnts[f"counts_{FV.U}"]
    cnts[f"counts_{FV.WD}"] = cnts[f"counts_{FV.U}"]

    if not add_uv:
        del dvrs[FV.U], dvrs[FV.V], cnts[f"counts_{FV.U}"], cnts[f"counts_{FV.V}"]
    if add_counts:
        dvrs.update(cnts)

    if wd_histo is not None:
        if verbosity > 0:
            print("Computing main wind direction")
        vname_binw = f"{vname_main_wd}_bin_width"
        i = 0
        width = 0
        dvrs[vname_main_wd] = np.full_like(dvrs[FV.WD], np.nan)
        dvrs[vname_binw] = np.zeros_like(dvrs[FV.WD])
        highest_density = np.zeros_like(dvrs[FV.WD])
        while width + wd_histo_minwidth <= wd_histo_maxwidth:
            width += wd_histo_minwidth
            n_bins = 2 * int(180 / width * 2)
            width = 360 / n_bins

            i += 1
            maxd = 0.0
            for b in range(n_bins):
                dens = wd_histo.roll(wd_histo, -b, axis=-1)
                dens = np.sum(dens[..., :i], axis=-1) + np.sum(dens[..., -i:], axis=-1)
                dens /= width
                maxd = max(maxd, np.max(dens))
                sel = dens > highest_density
                if np.any(sel):
                    highest_density[sel] = dens[sel]
                    dvrs[vname_main_wd][sel] = b * width
                    dvrs[vname_binw][sel] = width
                del dens, sel
            if verbosity > 1:
                print(
                    f"  bin width = {width:.2f} degrees: Max density = {maxd:.4e} counts/degree"
                )
        del highest_density

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
        "-mws",
        "--mean_ws_name",
        help="Output variable name for mean wind speed",
        default=FV.MEAN_WS,
    )
    parser.add_argument(
        "-mwd",
        "--main_wd_name",
        help="Output variable name for main wind direction",
        default=FV.MAIN_WD,
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
        "-uv",
        "--add_uv",
        help="Add U and V to the output",
        action="store_true",
    )
    parser.add_argument(
        "-cts",
        "--add_counts",
        help="Add counts for each variable",
        action="store_true",
    )
    parser.add_argument(
        "-wdm",
        "--wd_histo_minwidth",
        help="The minimal bin width of the main wind direction histogram",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-wdM",
        "--wd_histo_maxwidth",
        help="The maximal bin width of the main wind direction histogram",
        type=float,
        default=30.0,
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
            vname_mean_ws=args.mean_ws_name,
            vname_main_wd=args.main_wd_name,
            wd_histo_minwidth=args.wd_histo_minwidth,
            wd_histo_maxwidth=args.wd_histo_maxwidth,
            add_uv=args.add_uv,
            add_counts=args.add_counts,
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
