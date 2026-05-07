import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from xarray import open_dataset, Dataset, concat

from foxes.config import config
from foxes.core import Engine, get_engine
from foxes.utils import write_nc, calc_era5_density, ustar2ti
import foxes.variables as FV
import foxes.constants as FC


def _process_first_file(
    fpath,
    cmap,
    var2ncvar,
    lon_bounds=None,
    lat_bounds=None,
    preprocess=None,
    points_png=None,
    verbosity=1,
):
    """Process the first file to find variables and dimensions"""

    # read file:
    data = open_dataset(fpath, engine=config.nc_engine)
    if preprocess is not None:
        data = preprocess(data)

    # check variables:
    assert FV.U in var2ncvar and FV.V in var2ncvar, (
        f"var2ncvar must contain mappings for {FV.U} and {FV.V}, got: {var2ncvar}"
    )
    assert FV.WS not in var2ncvar and FV.WD not in var2ncvar, (
        f"var2ncvar should not contain mappings for {FV.WS} and {FV.WD}, as they will be computed from {FV.U} and {FV.V}"
    )
    assert var2ncvar[FV.U].endswith("100"), (
        f"Expected U variable to be at 100m height, got {var2ncvar[FV.U]}"
    )
    assert var2ncvar[FV.V].endswith("100"), (
        f"Expected V variable to be at 100m height, got {var2ncvar[FV.V]}"
    )
    c_msl = var2ncvar["msl"]
    assert c_msl in data.data_vars, (
        f"Mean sea level pressure variable '{c_msl}' is required to compute air density, but not found in data variables: {list(data.data_vars.keys())}"
    )
    c_t2m = var2ncvar["t2m"]
    assert c_t2m in data.data_vars, (
        f"2m temperature variable '{c_t2m}' is required to compute air density, but not found in data variables: {list(data.data_vars.keys())}"
    )
    c_d2m = var2ncvar["d2m"]
    assert c_d2m in data.data_vars, (
        f"2m dew point variable '{c_d2m}' is required to compute air density, but not found in data variables: {list(data.data_vars.keys())}"
    )
    c_zust = var2ncvar["zust"]
    assert c_zust in data.data_vars, (
        f"Ustar variable '{c_zust}' is required to compute TI, but not found in data variables: {list(data.data_vars.keys())}"
    )

    # reduce variables:
    ncvars = set(data.data_vars.keys())
    drop_vars = sorted(list(ncvars - set(var2ncvar.values())))
    keep_vars = [v for v in data.keys() if v not in drop_vars]
    if verbosity > 1:
        print(f"Found {len(ncvars)} variables in the files")
        print(f"Dropping {len(drop_vars)} variables: {drop_vars}")
        print(f"Keeping {len(keep_vars)} variables: {keep_vars}")
    data = data[keep_vars]
    for v in list(var2ncvar.keys()):
        if var2ncvar[v] not in keep_vars:
            del var2ncvar[v]

    # read coordinates:
    try:
        coords = {c: data.coords[nc].values for c, nc in cmap.items()}
    except KeyError as e:
        print(
            f"\nMissing coordinate in ERA5 data. Require {list(cmap.values())}, found: {list(data.coords.keys())}\n"
        )
        raise e
    if verbosity > 2:
        print("Found coordinates:")
        for c, vals in coords.items():
            print(f"  {c}: {vals.shape}, range: {np.min(vals)} - {np.max(vals)}")

    # apply bounds:
    points_isel = {}
    lon = coords[FV.LON]
    lat = coords[FV.LAT]
    if lon_bounds is not None:
        if verbosity > 1:
            print(f"Applying lon_bounds = {lon_bounds}")
        sel = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
        if not np.any(sel):
            raise ValueError(
                f"After applying longitude bounds {lon_bounds}, no points remain. Found longitude range: {np.min(lon)} - {np.max(lon)}. Check the bounds and the coordinate values."
            )
        lon = lon[sel]
        points_isel[FV.LON] = np.where(sel)[0]
        if verbosity > 2:
            print(
                f"Selecting {len(points_isel[FV.LON])} longitudes: {np.min(lon)} - {np.max(lon)}"
            )
    if lat_bounds is not None:
        if verbosity > 1:
            print(f"Applying lat_bounds = {lat_bounds}")
        sel = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
        if not np.any(sel):
            raise ValueError(
                f"After applying latitude bounds {lat_bounds}, no points remain. Found latitude range: {np.min(lat)} - {np.max(lat)}. Check the bounds and the coordinate values."
            )
        lat = lat[sel]
        points_isel[FV.LAT] = np.where(sel)[0]
        if verbosity > 2:
            print(
                f"Selecting {len(points_isel[FV.LAT])} latitudes: {np.min(lat)} - {np.max(lat)}"
            )

    # write grid points plot:
    if points_png is not None:
        if verbosity > 0:
            print(f"Saving grid points plot to {points_png}")
        lonlat = np.meshgrid(lon, lat)
        plt.figure(figsize=(7, 7))
        plt.scatter(lonlat[0], lonlat[1], s=1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("ERA5 Grid Points")
        plt.tight_layout()
        plt.savefig(points_png)
        plt.close()

    return drop_vars, points_isel


def _process_file(
    fpath,
    cmap,
    var2ncvar,
    drop_vars,
    points_isel,
    preprocess=None,
    check_nan=True,
    max_ti=1.0,
    verbosity=0,
):
    """Process a single file"""
    # prepare:
    cs = cmap[FC.STATE]
    clon = cmap[FV.LON]
    clat = cmap[FV.LAT]
    c_u = var2ncvar[FV.U]
    c_v = var2ncvar[FV.V]
    c_msl = var2ncvar["msl"]
    c_t2m = var2ncvar["t2m"]
    c_d2m = var2ncvar["d2m"]
    c_zust = var2ncvar["zust"]
    with open_dataset(
        fpath, drop_variables=drop_vars, engine=config.nc_engine
    ) as era5_data:
        if preprocess is not None:
            era5_data = preprocess(era5_data)
        if FV.LON in points_isel:
            era5_data = era5_data.isel({clon: points_isel[FV.LON]})
        if FV.LAT in points_isel:
            era5_data = era5_data.isel({clat: points_isel[FV.LAT]})
        era5_data.load()

    # extract times:
    times = era5_data[cs].values
    years = np.unique(times.astype("datetime64[Y]").astype(int) + 1970)
    months = np.unique(times.astype("datetime64[M]").astype(int) % 12 + 1)
    # days = (times.astype('datetime64[D]') - times.astype('datetime64[M]')).astype(int) + 1
    assert len(years) == 1, f"Expected all times to be in the same year, found: {years}"
    assert len(months) == 1, (
        f"Expected all times to be in the same month, found: {months}"
    )
    year = years[0]
    month = months[0]
    del times, years, months

    # extract data:
    ocmap = {FC.STATE: "Time", FV.LAT: FV.LAT, FV.LON: FV.LON}
    crds = {ocmap[c]: era5_data.coords[nc].values for c, nc in cmap.items()}
    data = []
    for w in var2ncvar.values():
        assert era5_data[w].dims == (cs, clat, clon), (
            f"Expected dimensions ({cs}, {clat}, {clon}) for variable {w}, found {era5_data[w].dims}"
        )
        data.append(era5_data[w].values)

    # compute air density:
    data.append(calc_era5_density(era5_data, z=100.0, var2ncvar=var2ncvar))
    vrs = list(var2ncvar.keys()) + [FV.RHO]
    for v in [c_msl, c_t2m, c_d2m]:
        if v in vrs:
            i = vrs.index(v)
            data.pop(i)
            vrs.pop(i)

    # compute ti:
    ws = np.sqrt(era5_data[c_u].values ** 2 + era5_data[c_v].values ** 2)
    ustar = era5_data[c_zust].values
    data.append(ustar2ti(ustar, ws, max_ti=max_ti))
    vrs.append(FV.TI)
    if c_zust in vrs:
        i = vrs.index(c_zust)
        data.pop(i)
        vrs.pop(i)
    del era5_data, ws, ustar

    # check for nan values:
    if check_nan:
        for i, w in enumerate(vrs):
            n_nan = np.sum(np.isnan(data[i]))
            if n_nan > 0:
                raise ValueError(
                    f"{fpath.name}: Found {n_nan} NaN values in variable {w}"
                )

    # create Dataset:
    data = Dataset(
        coords=crds,
        data_vars={v: (tuple(ocmap.values()), data[i]) for i, v in enumerate(vrs)},
        attrs={
            "source_file": fpath.name,
            f"height_{FV.U}": 100.0,
            f"height_{FV.V}": 100.0,
        },
    )

    return data, f"{year:04d}{month:02d}"


def _write_file(data, fpath, write_pars=None, verbosity=0):
    """Write the processed data to a NetCDF file"""
    wpars = dict(pack=True)
    if write_pars is not None:
        wpars.update(write_pars)
    write_nc(data, fpath, verbosity=verbosity, **wpars)


def era52foxes(
    source_files,
    out_dir,
    cmap=None,
    var2ncvar=None,
    lon_bounds=None,
    lat_bounds=None,
    preprocess=None,
    write_points_png=False,
    check_nan=False,
    write_pars=None,
    max_ti=1.0,
    verbosity=1,
):
    """
    Convert ERA5 NetCDF files to the foxes format expected by
    the FieldData states class.

    Parameters
    ----------
    source_files : str
        Source files to process, either a single file or a glob pattern.
    out_dir : str
        Output directory for resulting NetCDF files.
    cmap: dict, optional
        Mapping from foxes dimension name to ERA5 dimension name
    var2ncvar: dict, optional
        Mapping from foxes variable to ERA5 variable name
    lon_bounds: tuple, optional
        The longitude bounds (min, max) to subset the data, in degrees
    lat_bounds: tuple, optional
        The latitude bounds (min, max) to subset the data, in degrees
    preprocess: function, optional
        A function that takes the opened ERA5 dataset and returns a modified dataset,
    write_points_png: bool, optional
        Whether to save a plot of the grid points
    write_pars: dict, optional
        Parameters for writing the NetCDF file, e.g. pack
    max_ti: float, optional
        The maximum turbulence intensity (TI) value to compute
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
        FC.STATE: "valid_time",
        FV.LAT: "latitude",
        FV.LON: "longitude",
    }
    if cmap is not None:
        cmap.update(cmap)
    var2ncvar = {
        FV.U: "u100",
        FV.V: "v100",
        "msl": "msl",
        "t2m": "t2m",
        "d2m": "d2m",
        "zust": "zust",
    }
    if var2ncvar is not None:
        var2ncvar.update(var2ncvar)

    # find variables to drop:
    if verbosity > 0:
        print(f"Preprocessing file {files[0].name}")
    drop_vars, points_isel = _process_first_file(
        files[0],
        cmap,
        var2ncvar,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        preprocess=preprocess,
        points_png=points_png,
        verbosity=verbosity,
    )

    # submit to workers:
    futures = [
        engine.submit(
            _process_file,
            fpath=fpath,
            cmap=cmap,
            var2ncvar=var2ncvar,
            drop_vars=drop_vars,
            preprocess=preprocess,
            points_isel=points_isel,
            check_nan=check_nan,
            max_ti=max_ti,
            verbosity=verbosity - 2,
        )
        for fpath in files
    ]

    ym = None
    data = []
    done = 0
    proc = -1
    total = len(futures)
    while len(futures) > 0:
        ds, hym = engine.await_result(futures.pop(0))
        data.append(ds)

        if ym is None:
            ym = hym
        elif hym != ym or len(futures) == 0:
            fpath = out_dir / f"ERA5_{ym}.nc"
            _write_file(
                data=concat(data, dim="Time", join="exact"),
                fpath=fpath,
                write_pars=write_pars,
                verbosity=verbosity - 1,
            )
            data = []
            ym = hym

        done += 1
        if verbosity > 0:
            hproc = int(100 * (done / total))
            if hproc > proc:
                proc = hproc
                print(f"Progress: {proc}% ({done}/{total} files processed)")


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
        "-mti",
        "--max_ti",
        help="The maximum turbulence intensity (TI) value to compute",
        type=float,
        default=1.0,
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
        return era52foxes(
            source_files=args.source_files,
            out_dir=args.out_dir,
            lon_bounds=args.lon_bounds,
            lat_bounds=args.lat_bounds,
            write_points_png=args.write_points_png,
            check_nan=not args.skip_check_nan,
            write_pars=dict(pack=not args.skip_packing),
            max_ti=args.max_ti,
            verbosity=args.verbosity,
        )


if __name__ == "__main__":
    main()
