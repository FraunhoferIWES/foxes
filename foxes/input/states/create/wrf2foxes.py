import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.autonotebook import tqdm
from xarray import open_dataset, DataArray, Dataset
from utm import latlon_to_zone_number, latitude_to_zone_letter, from_latlon, to_latlon
from scipy.interpolate import (
    griddata,
    LinearNDInterpolator,
    NearestNDInterpolator,
    CloughTocher2DInterpolator,
)

from foxes.config import config
from foxes.core import Engine, get_engine
from foxes.utils import wd2uv, uv2wd, write_nc
import foxes.variables as FV
import foxes.constants as FC


def _process_first_file(
    fpath,
    cmap,
    var2ncvar,
    resolution=None,
    lon_bounds=None,
    lat_bounds=None,
    height_bounds=None,
    preprocess=None,
    points_png=None,
    verbosity=1,
):
    """Process the first file to find variables and dimensions"""

    # read file:
    wrf_data = open_dataset(fpath, engine=config.nc_engine)
    if preprocess is not None:
        wrf_data = preprocess(wrf_data)

    # reduce variables:
    ncvars = set(wrf_data.data_vars.keys())
    drop_vars = sorted(list(ncvars - set(var2ncvar.values())))
    keep_vars = [v for v in wrf_data.keys() if v not in drop_vars]
    if verbosity > 1:
        print(f"Found {len(ncvars)} variables in the files")
        print(f"Dropping {len(drop_vars)} variables: {drop_vars}")
        print(f"Keeping {len(keep_vars)} variables: {keep_vars}")
    wrf_data = wrf_data[keep_vars]
    for v in list(var2ncvar.keys()):
        if var2ncvar[v] not in keep_vars:
            del var2ncvar[v]

    # read coordinates:
    try:
        coords = {c: wrf_data.coords[nc].values for c, nc in cmap.items()}
    except KeyError as e:
        print(
            f"\nMissing coordinate in WRF data. Require {list(cmap.values())}, found: {list(wrf_data.coords.keys())}\n"
        )
        raise e
    if verbosity > 2:
        print("Found coordinates:")
        for c, vals in coords.items():
            print(f"  {c}: {vals.shape}, range: {np.min(vals)} - {np.max(vals)}")

    # apply bounds:
    if lon_bounds is not None or lat_bounds is not None:
        if verbosity > 1:
            if lon_bounds is not None:
                print(f"Applying lon_bounds = {lon_bounds}")
            if lat_bounds is not None:
                print(f"Applying lat_bounds = {lat_bounds}")
        lon_bounds = (0.0, 360.0) if lon_bounds is None else lon_bounds
        lat_bounds = (-90.0, 90.0) if lat_bounds is None else lat_bounds
        lon = coords[FV.LON]
        lat = coords[FV.LAT]
        sel = (
            (lon >= lon_bounds[0])
            & (lon <= lon_bounds[1])
            & (lat >= lat_bounds[0])
            & (lat <= lat_bounds[1])
        )
        assert np.any(sel), (
            f"After applying bounds lon_bounds = {lon_bounds}, lat_bounds = {lat_bounds}, no points remain. Found longitude range: {np.min(lon)} - {np.max(lon)}, latitude range: {np.min(lat)} - {np.max(lat)}. Check the bounds and the coordinate values."
        )
        lon = lon[sel]
        lat = lat[sel]
        assert wrf_data[cmap[FV.LON]].dims == (cmap[FV.Y], cmap[FV.X]) and wrf_data[
            cmap[FV.LAT]
        ].dims == (cmap[FV.Y], cmap[FV.X]), (
            f"Expected longitude and latitude coordinates to have dimensions ({cmap[FV.Y]}, {cmap[FV.X]}), found {wrf_data[cmap[FV.LON]].dims} and {wrf_data[cmap[FV.LAT]].dims}"
        )
        sel = np.where(sel)
        points_isel = {FV.X: sel[1], FV.Y: sel[0]}
        if verbosity > 2:
            print(
                f"Selecting {len(points_isel[FV.X])} points: lon {np.min(lon)} - {np.max(lon)}, lat {np.min(lat)} - {np.max(lat)}"
            )
    else:
        nx, ny = coords[FV.LON].shape
        lon = coords[FV.LON].reshape(nx * ny)
        lat = coords[FV.LAT].reshape(nx * ny)
        points_isel = {}
    if height_bounds is not None:
        if verbosity > 1:
            print(f"Applying height_bounds = {height_bounds}")
        h = coords[FV.H]
        sel = (h >= height_bounds[0]) & (h <= height_bounds[1])
        assert np.any(sel), (
            f"After applying height bounds {height_bounds}, no points remain. Found height range: {np.min(h)} - {np.max(h)}. Check the bounds and the coordinate values."
        )
        h = h[sel]
        points_isel[FV.H] = np.where(sel)[0]
        if verbosity > 2:
            print(
                f"Selecting {len(points_isel[FV.H])} heights: {np.min(h)} - {np.max(h)}"
            )

    # find resolution:
    if resolution is None:
        xstep = np.round(np.mean(coords[FV.X][1:] - coords[FV.X][:-1]), 2)
        ystep = np.round(np.mean(coords[FV.Y][1:] - coords[FV.Y][:-1]), 2)
        resolution = np.min([xstep, ystep])
        if verbosity > 2:
            print(f"Found grid steps: xstep={xstep}, ystep={ystep}")
    if verbosity > 0:
        print(f"Using resolution    : {resolution} m")

    # find UTM zone:
    utm_zone_number = latlon_to_zone_number(np.mean(lat), np.mean(lon))
    utm_zone_letter = latitude_to_zone_letter(np.mean(lat))
    utm_zone = (utm_zone_number, utm_zone_letter)
    if verbosity > 0:
        print(f"Using UTM zone      : {utm_zone_number}{utm_zone_letter}")

    # find grid points:
    wrf_points = from_latlon(
        lat, lon, force_zone_number=utm_zone_number, force_zone_letter=utm_zone_letter
    )
    wrf_points = np.stack(wrf_points[:2], axis=-1)
    interp = LinearNDInterpolator(
        wrf_points, np.arange(wrf_points.shape[0]), fill_value=np.nan
    )
    p_min = np.min(wrf_points, axis=0)
    p_max = np.max(wrf_points, axis=0)
    step = resolution / 10
    while p_max[0] - p_min[0] >= resolution and p_max[1] - p_min[1] >= resolution:
        x = np.arange(p_min[0], p_max[0] + resolution, resolution)
        y = np.arange(p_min[1], p_max[1] + resolution, resolution)
        nx = len(x)
        ny = len(y)
        points = np.zeros((nx, ny, 2), dtype=config.dtype_double)
        points[:, :, 0] = x[:, None]
        points[:, :, 1] = y[None, :]
        ok = True
        if np.any(np.isnan(interp(points[0, :, :]))):
            p_min[0] += step
            ok = False
        if np.any(np.isnan(interp(points[-1, :, :]))):
            p_max[0] -= step
            ok = False
        if np.any(np.isnan(interp(points[:, 0, :]))):
            p_min[1] += step
            ok = False
        if np.any(np.isnan(interp(points[:, -1, :]))):
            p_max[1] -= step
            ok = False
        if ok:
            break
    del interp
    if p_max[0] - p_min[0] < resolution or p_max[1] - p_min[1] < resolution:
        raise ValueError(
            f"Cannot satisfy bounds lon={lon_bounds}, lat={lat_bounds} with resolution {resolution} m. Final bounds: {p_min} - {p_max}"
        )
    if verbosity > 1:
        print(f"Found grid points   : nx={nx}, ny={ny}, total={nx * ny}")
    if verbosity > 2:
        ll = to_latlon(
            points[:, :, 0], points[:, :, 1], utm_zone_number, utm_zone_letter
        )
        if lon_bounds is not None:
            print(f"Target lon range    : {lon_bounds[0]:.8f} - {lon_bounds[1]:.8f}")
            print(f"Realized lon range  : {np.min(ll[1]):.8f} - {np.max(ll[1]):.8f}")
        if lat_bounds is not None:
            print(f"Target lat range    : {lat_bounds[0]:.8f} - {lat_bounds[1]:.8f}")
            print(f"Realized lat range  : {np.min(ll[0]):.8f} - {np.max(ll[0]):.8f}")
        if height_bounds is not None:
            print(
                f"Initial height range: {np.min(coords[FV.H]):.2f} - {np.max(coords[FV.H]):.2f}"
            )
            print(f"Final height range  : {np.min(h):.2f} - {np.max(h):.2f}")
        del ll

    # write grid points plot:
    if points_png is not None:
        if verbosity > 0:
            print(f"Saving grid points plot to {points_png}")
        ll = to_latlon(
            points[:, :, 0], points[:, :, 1], utm_zone_number, utm_zone_letter
        )
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(lon, lat, s=1, label="WRF points")
        plt.scatter(ll[1], ll[0], s=1, label="Grid points")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(wrf_points[:, 0], wrf_points[:, 1], s=1, label="WRF points")
        plt.scatter(points[:, :, 0], points[:, :, 1], s=1, label="Grid points")
        plt.xlabel("UTM X")
        plt.ylabel("UTM Y")
        plt.title(
            f"UTM zone: {utm_zone_number}{utm_zone_letter}, grid spacing: {resolution} m"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(points_png)
        plt.close()

    interp_data = (wrf_points, points.reshape(nx, ny, 2))

    return drop_vars, utm_zone, points_isel, interp_data


def _process_file(
    fpath,
    cmap,
    var2ncvar,
    out_dir,
    drop_vars,
    utm_zone,
    points_isel,
    interp_data,
    chunk_size_states=None,
    chunk_size_points=None,
    preprocess=None,
    check_nan=True,
    interp_pars=None,
    write_pars=None,
    verbosity=0,
):
    """Process a single file"""
    # prepare:
    cs = cmap[FC.STATE]
    cx = cmap[FV.X]
    cy = cmap[FV.Y]
    ch = cmap[FV.H]

    with open_dataset(
        fpath, drop_variables=drop_vars, engine=config.nc_engine
    ) as wrf_data:
        if preprocess is not None:
            wrf_data = preprocess(wrf_data)
        if FV.H in points_isel:
            wrf_data = wrf_data.isel({ch: points_isel[FV.H]})
        wrf_data = wrf_data.load()

    # extract data:
    n_times = wrf_data.sizes[cs]
    if FV.X in points_isel:
        assert FV.Y in points_isel, (
            f"If {FV.X} is in points_isel, {FV.Y} must also be in points_isel"
        )
        x = DataArray(wrf_data[cx].values[points_isel[FV.X]], dims="points")
        y = DataArray(wrf_data[cy].values[points_isel[FV.Y]], dims="points")
        n_points = x.sizes["points"]
        wrf_data = wrf_data.sel({cx: x, cy: y}, method="nearest")
        data = [wrf_data[var2ncvar[v]].values for v in var2ncvar.keys()]
        dims = [wrf_data[var2ncvar[v]].dims for v in var2ncvar.keys()]
        del x, y
    else:
        nx, ny = wrf_data.sizes[cx], wrf_data.sizes[cy]
        n_points = nx * ny
        data = []
        dims = []
        for w in var2ncvar.values():
            if cx in wrf_data[w].dims or cy in wrf_data[w].dims:
                assert wrf_data[w].dims[-2:] == (cy, cx), (
                    f"Expected variable {w} to have dimensions ending with ({cy}, {cx}), found {wrf_data[w].dims}"
                )
                shp = wrf_data[w].shape
                data.append(wrf_data[w].values.reshape(shp[:-2] + (n_points,)))
                dims.append(wrf_data[w].dims[:-2] + ("points",))
            else:
                data.append(wrf_data[w].values)
                dims.append(wrf_data[w].dims)
    if check_nan:
        for i, w in enumerate(var2ncvar.values()):
            n_nan = np.sum(np.isnan(data[i]))
            if n_nan > 0:
                raise ValueError(
                    f"{fpath.name}: Found {n_nan} NaN values in variable {w}"
                )
    times = wrf_data[cs].values
    heights = wrf_data[ch].values if ch in wrf_data else None
    del wrf_data

    # sort data according to dimensions:
    vrs = {
        dms: [v for v, dm in zip(var2ncvar.keys(), dims) if dm == dms]
        for dms in set(dims)
    }
    data = {
        dms: np.stack([d for d, dm in zip(data, dims) if dm == dms], axis=-1)
        for dms in set(dims)
    }

    # replace WS, WD by U, V:
    found_ws_wd = False
    for dms in vrs:
        hvrs = vrs[dms]
        if FV.WS in hvrs or FV.WD in hvrs:
            assert FV.WS in hvrs and FV.WD in hvrs, (
                f"Both {FV.WS} and {FV.WD} must be present, found {hvrs} for dimensions {dms}"
            )
            iws = iu = hvrs.index(FV.WS)
            iwd = iv = hvrs.index(FV.WD)
            uv = wd2uv(data[dms][..., iwd], data[dms][..., iws])
            data[dms][..., iu] = uv[..., 0]
            data[dms][..., iv] = uv[..., 1]
            hvrs[iu] = FV.U
            hvrs[iv] = FV.V
            found_ws_wd = True
            break
    assert found_ws_wd, (
        f"Did not find both {FV.WS} and {FV.WD} in the data, cannot compute {FV.U} and {FV.V}"
    )

    # define output coordinate names:
    icmap = {nc: c for c, nc in cmap.items()}
    ocmap = {v: v for v in cmap.keys()}
    ocmap.update(
        {
            FC.STATE: "Time",
            FV.X: "UTMX",
            FV.Y: "UTMY",
            FV.H: "height",
        }
    )

    # reorder dimensions:
    vrs_list = []
    for dms in vrs.keys():
        if cs in dms:
            assert dms[0] == cs, (
                f"Expected dimension {cs} to be the first dimension in {dms} for variables {vrs[dms]}"
            )
        if "points" in dms:
            assert dms[-1] == "points", (
                f"Expected dimension 'points' to be the last dimension in {dms} for variables {vrs[dms]}"
            )
            odms = [c for c in ("points", cs, ch) if c in dms]
            odms += [c for c in dms if c not in odms]
            data[dms] = np.moveaxis(
                data[dms], [dms.index(c) for c in odms], range(len(odms))
            )
            vrs_list.append(tuple(odms))

        else:
            vrs_list.append(dms)
    vrs = {vrs_list[i]: d for i, d in enumerate(vrs.values())}
    data = {vrs_list[i]: d for i, d in enumerate(data.values())}

    # prepare interpolation:
    ipars = dict(method="linear", rescale=True)
    if interp_pars is not None:
        ipars.update(interp_pars)

    def _interpolate(pts, arr, qts):
        if not check_nan:
            s = np.any(np.isnan(arr), axis=tuple(range(1, len(arr.shape))))
            s = np.s_[~s, ...]
            pts = pts[s]
            arr = arr[s]
            del s

        if chunk_size_points is None:
            return griddata(pts, arr, qts, **ipars)
        else:
            hpars = {k: d for k, d in ipars.items() if k != "method"}
            if ipars["method"] == "nearest":
                interp = NearestNDInterpolator(pts, arr, **hpars)
            elif ipars["method"] == "linear":
                interp = LinearNDInterpolator(pts, arr, **hpars)
            elif ipars["method"] == "cubic":
                interp = CloughTocher2DInterpolator(pts, arr, **hpars)
            else:
                raise ValueError(
                    f"Unsupported interpolation method {ipars['method']}, supported methods are 'linear', 'nearest', and 'cubic'"
                )

            nx, ny = qts.shape[:2]
            n_points = nx * ny
            qts = qts.reshape(n_points, 2)
            done_points = 0
            res = []
            while done_points < n_points:
                p_chunk = slice(
                    done_points, min(done_points + chunk_size_points, n_points)
                )
                if verbosity > 3:
                    print(
                        f"  {fpath.name}: INTERPOLATING {dms}, {hvrs}, done_points {done_points}/{n_points}, p_chunk {p_chunk}"
                    )
                res.append(interp(qts[p_chunk]))
                done_points += p_chunk.stop - p_chunk.start
            res = np.concatenate(res, axis=0)
            return res.reshape(nx, ny, *res.shape[1:])

    # interpolate to grid:
    wrf_points, grid_points = interp_data
    nx, ny = grid_points.shape[:2]
    done_times = 0
    temp_data = data
    crds = {}
    data = {}
    while done_times < n_times:
        t_chunk = (
            slice(done_times, min(done_times + chunk_size_states, n_times))
            if chunk_size_states is not None
            else slice(None)
        )
        ctimes = np.arange(n_times)[t_chunk]
        nc = len(ctimes)

        for dms, arr in temp_data.items():
            hvrs = tuple(vrs[dms])
            if verbosity > 2:
                print(
                    f"{fpath.name}: INTERPOLATING {dms}, {hvrs}, done_times {done_times}/{n_times}, t_chunk {t_chunk}"
                )

            if cs in dms and icmap[cs] not in crds:
                crds[ocmap[FC.STATE]] = times
            if "points" in dms and icmap[cx] not in crds:
                crds[ocmap[FV.X]] = grid_points[:, 0, 0]
                crds[ocmap[FV.Y]] = grid_points[0, :, 1]
            if ch in dms and icmap[ch] not in crds:
                crds[ocmap[FV.H]] = heights

            if cs in dms:
                arr = arr[:, t_chunk, ...]
            if "points" in dms:
                if cs in dms or done_times == 0:
                    res = _interpolate(wrf_points, arr, grid_points)
                    dms = (FV.X, FV.Y) + tuple([icmap[c] for c in dms[1:]])
                else:
                    continue
            elif cs in dms or done_times == 0:
                res = arr
                dms = tuple([icmap[c] for c in dms])
            else:
                continue

            if hvrs not in data.keys():
                data[hvrs] = [dms, [res]]
            else:
                data[hvrs][1].append(res)
            del res, dms

        done_times += nc
    del temp_data

    # recombine chunks:
    for hvrs in data.keys():
        if len(data[hvrs][1]) > 1:
            data[hvrs][1] = np.concatenate(data[hvrs][1], axis=2)
        else:
            data[hvrs][1] = data[hvrs][1][0]

    # organize data for Dataset, and replace U, V by WS, WD:
    for hvrs in list(data.keys()):
        (dms, arr) = data.pop(hvrs)
        hvrs = list(hvrs)
        if FV.U in hvrs:
            uv = np.stack([arr[..., iu], arr[..., iv]], axis=-1)
            arr[..., iws] = np.linalg.norm(uv, axis=-1)
            arr[..., iwd] = uv2wd(uv)
            hvrs[iws] = FV.WS
            hvrs[iwd] = FV.WD
            del uv
        odms = [c for c in [FC.STATE, FV.H, FV.Y, FV.X] if c in dms]
        odmi = [dms.index(c) for c in odms]
        odms = [ocmap[c] for c in odms] + [ocmap[c] for c in dms if c not in odms]
        arr = np.moveaxis(arr, odmi, range(len(odmi)))
        data.update({v: (tuple(odms), arr[..., i]) for i, v in enumerate(hvrs)})

    # create Dataset:
    data = Dataset(
        coords=crds,
        data_vars=data,
        attrs={
            "source_file": fpath.name,
            "utm_number": utm_zone[0],
            "utm_letter": utm_zone[1],
        },
    )

    # write file:
    wpars = dict(pack=True)
    if write_pars is not None:
        wpars.update(write_pars)
    out_path = out_dir / f"{fpath.stem}_UTM{utm_zone[0]}{utm_zone[1]}.nc"
    write_nc(data, out_path, verbosity=verbosity, **wpars)


def wrf2foxes(
    source_files,
    out_dir,
    cmap=None,
    var2ncvar=None,
    resolution=None,
    lon_bounds=None,
    lat_bounds=None,
    height_bounds=(0.0, 400.0),
    chunk_size_states=None,
    chunk_size_points=None,
    preprocess=None,
    write_points_png=False,
    check_nan=False,
    interp_pars=None,
    write_pars=None,
    verbosity=1,
):
    """
    Convert WRF NetCDF files to the foxes format expected by
    the FieldData states class.

    Parameters
    ----------
    source_files : str
        Source files to process, either a single file or a glob pattern.
    out_dir : str
        Output directory for resulting NetCDF files.
    cmap: dict, optional
        Mapping from foxes dimension name to WRF dimension name
    var2ncvar: dict, optional
        Mapping from foxes variable to WRF variable name
    resolution: float, optional
        The grid resolution in m, if not provided, it will be determined
    lon_bounds: tuple, optional
        The longitude bounds (min, max) to subset the data, in degrees
    lat_bounds: tuple, optional
        The latitude bounds (min, max) to subset the data, in degrees
    height_bounds: tuple, optional
        The height bounds (min, max) to subset the data, in meters
    chunk_size_states: int, optional
        The chunk size for time dimension during interpolation
    chunk_size_points: int, optional
        The chunk size for target points during interpolation
    preprocess: function, optional
        A function that takes the opened WRF dataset and returns a modified dataset,
    write_points_png: bool, optional
        Whether to save a plot of the grid points
    interp_pars: dict, optional
        Parameters for the interpolation via griddata,
        e.g. method, rescale, fill_value
    write_pars: dict, optional
        Parameters for writing the NetCDF file, e.g. pack
    verbosity : int, optional
        The verbosity level, 0 = silent, by default 1

    """

    # prepare:
    engine = get_engine()
    source_files = Path(source_files)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    points_png = out_dir / "grid_points.png" if write_points_png else None

    # find files:
    prt = source_files.resolve().parent
    glb = source_files.name
    while "*" in str(prt):
        glb = prt.name + "/" + glb
        prt = prt.parent
    files = sorted(list(prt.glob(glb)))
    if verbosity > 0:
        print(f"Found {len(files)} files to process.")

    # variable mapping:
    cmap = {
        FC.STATE: "time",
        FV.H: "height",
        FV.Y: "south_north",
        FV.X: "west_east",
        FV.LAT: "XLAT",
        FV.LON: "XLON",
    }
    if cmap is not None:
        cmap.update(cmap)
    var2ncvar = {
        FV.WS: "WS",
        FV.WD: "WD",
        FV.TKE: "TKE",
        FV.RHO: "RHO",
        FV.T: "T",
    }
    if var2ncvar is not None:
        var2ncvar.update(var2ncvar)

    # find variables to drop:
    if verbosity > 0:
        print(f"Preprocessing file {files[0].name}")
    drop_vars, utm_zone, points_isel, interp_data = _process_first_file(
        files[0],
        cmap,
        var2ncvar,
        resolution=resolution,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        preprocess=preprocess,
        points_png=points_png,
        height_bounds=height_bounds,
        verbosity=verbosity,
    )

    # submit to workers:
    futures = [
        engine.submit(
            _process_file,
            fpath=fpath,
            cmap=cmap,
            var2ncvar=var2ncvar,
            out_dir=out_dir,
            drop_vars=drop_vars,
            preprocess=preprocess,
            utm_zone=utm_zone,
            points_isel=points_isel,
            interp_data=interp_data,
            chunk_size_states=chunk_size_states,
            chunk_size_points=chunk_size_points,
            check_nan=check_nan,
            interp_pars=interp_pars,
            write_pars=write_pars,
            verbosity=verbosity - 2,
        )
        for fpath in files
    ]

    if verbosity > 0:
        [
            engine.await_result(f)
            for f in tqdm(futures, desc="Processing files", unit="file")
        ]
    else:
        [engine.await_result(f) for f in futures]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_files",
        help="Source files to process, either a single file or a glob pattern",
        type=str,
    )
    parser.add_argument(
        "out_dir",
        help="Output directory for resulting NetCDF files",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        help="The grid resolution in m, if not provided, it will be determined",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-lob",
        "--lon_bounds",
        help="The longitude bounds (min, max) to subset the data, in degrees",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-lab",
        "--lat_bounds",
        help="The latitude bounds (min, max) to subset the data, in degrees",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-hb",
        "--height_bounds",
        help="The height bounds (min, max) to subset the data, in meters",
        type=float,
        nargs=2,
        default=(0.0, 400.0),
    )
    parser.add_argument(
        "-c",
        "--chunk_size_states",
        help="The chunk size for time dimension during interpolation",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-C",
        "--chunk_size_points",
        help="The chunk size for target points during interpolation",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="The engine",
        default="process",
    )
    parser.add_argument(
        "-n",
        "--n_cpus",
        help="The number of cpus",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-png",
        "--write_points_png",
        help="Whether to save a plot of the grid points",
        action="store_true",
    )
    parser.add_argument(
        "-im",
        "--interp_method",
        help="The interpolation method for griddata, e.g. 'linear', 'nearest', 'cubic'",
        type=str,
        default="linear",
    )
    parser.add_argument(
        "-sp",
        "--skip_packing",
        help="Skip packing the data when writing NetCDF files",
        action="store_true",
    )
    parser.add_argument(
        "-scn",
        "--skip_check_nan",
        help="Whether to skip the check for NaN values",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="The verbosity level, 0 = silent",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    with Engine.new(args.engine, n_procs=args.n_cpus):
        return wrf2foxes(
            source_files=args.source_files,
            out_dir=args.out_dir,
            resolution=args.resolution,
            lon_bounds=args.lon_bounds,
            lat_bounds=args.lat_bounds,
            height_bounds=args.height_bounds,
            write_points_png=args.write_points_png,
            chunk_size_states=args.chunk_size_states,
            chunk_size_points=args.chunk_size_points,
            check_nan=not args.skip_check_nan,
            interp_pars=dict(method=args.interp_method),
            write_pars=dict(pack=not args.skip_packing),
            verbosity=args.verbosity,
        )


if __name__ == "__main__":
    main()
