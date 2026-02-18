import argparse
import numpy as np
from xarray import Dataset
from pathlib import Path
from tqdm.autonotebook import tqdm

from foxes.core import get_engine, Engine
from foxes.config import config
from foxes.utils import write_nc, import_module, download_file
from foxes.data import StaticData, STATES
import foxes.variables as FV


def _get_file_var_str(var, for_fname=False):
    """Get the variable string for the filename based on the variable code."""
    var_str = {
        FV.U: "U",
        FV.V: "V",
        FV.TKE: "TKE",
        FV.p: "P",
        FV.T: "T",
        None: "",
    }[var]
    if for_fname and var is not None:
        var_str = f"_{var_str}"
    return var_str


def _get_fname(year, month, var=None, region=None, suffix="nc"):
    """Construct the filename for a given year, month, and variable."""
    ym_str = f"{year}{month:02d}"
    var_str = _get_file_var_str(var, for_fname=True)
    region_str = f"_{region}" if region is not None else ""
    return f"ICON-DREAM-EU_{ym_str}{region_str}{var_str}_hourly.{suffix}"


def _download_icon_dream(ymv, base_url, out_dir, verbosity=1):
    """Download a file from ICON-DREAM-EU for a given year, month, and variable."""
    year, month, var = ymv
    fname = _get_fname(year, month, var, region=None, suffix="grb")
    var_str = _get_file_var_str(var)
    url = f"{base_url}/{var_str}/{fname}"
    var_dir = out_dir / var_str
    var_dir.mkdir(parents=True, exist_ok=True)
    out_path = var_dir / fname
    return download_file(url, out_path, verbosity=verbosity)


def _prepare_grid(
    path_grid_select, path_icon_grid, path_weights_out, url_icon_grid, verbosity=1
):
    """Download and prepare grid files for remapping."""
    if path_weights_out.is_file():
        return 0  # Already present
    if download_file(url_icon_grid, path_icon_grid, verbosity=verbosity) < 0:
        return -1  # Indicate failure

    Cdo = import_module(
        "cdo",
        pip_hint="pip install cdo",
        conda_hint="conda install -c conda-forge cdo",
    ).Cdo
    cdo = Cdo()
    cdo.gencon(
        path_grid_select, input=f"{path_icon_grid}", output=str(path_weights_out)
    )
    return 1  # Indicate success


def _process(
    region,
    year,
    month,
    grb_dir,
    nc_dir,
    var2ncvar,
    levels,
    path_grid_select,
    path_grid_weights,
):
    """Process grb files and convert to NetCDF."""
    nc_fname = _get_fname(year, month, var=None, region=region, suffix="nc")
    nc_path = nc_dir / nc_fname
    if nc_path.exists():
        return 0  # Indicate already processed

    Cdo = import_module(
        "cdo",
        pip_hint="pip install cdo",
        conda_hint="conda install -c conda-forge cdo",
    ).Cdo
    cdo = Cdo()
    data = {}
    for var, vname in var2ncvar.items():
        grb_fname = _get_fname(year, month, var, region=None, suffix="grb")
        grb_path = grb_dir / _get_file_var_str(var) / grb_fname

        if not grb_path.exists():
            return -1  # Indicate failure

        # select levels:
        lvls = levels if var != "TKE" else levels + [levels[-1] + 1]
        lvls = ",".join(str(lv) for lv in lvls)
        temp = cdo.sellevel(lvls, input=str(grb_path), returnXArray=vname)

        # remap:
        data[var] = cdo.remap(
            str(path_grid_select), path_grid_weights, input=temp, returnXArray=vname
        )
        if var == "TKE":
            data[var] = data[var].rename({"height": "height_2"})

    data = Dataset(data)
    write_nc(data, nc_path, nc_engine=config.nc_engine, verbosity=0)
    return 1  # Indicate success


def iconDream2foxes(
    out_dir,
    region,
    min_year,
    min_month,
    max_year,
    max_month,
    base_url="https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-EU/hourly",
    url_icon_grid="http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0027_R03B08_N02.nc",
    levels=None,
    verbosity=1,
):
    """
    Download ICON-DREAM-EU hourly files for specified variables and time range,
    and convert them into foxes compatible NetCDF files.

    Parameters
    ----------
    out_dir: str or Path
        Directory to save downloaded files.
    region: str
        Region for which to download data ("northsea" or "baltic").
    min_year: int
        Minimal year (inclusive).
    min_month: int
        Minimal month (inclusive).
    max_year: int
        Maximal year (inclusive).
    max_month: int
        Maximal month (inclusive).
    base_url: str
        Base URL of the FTP server.
    url_icon_grid: str
        URL to download the ICON grid file if not present.
    levels: list of int, optional
        The ICON height levels, e.g. [69,70,71,72,73,74].
    verbosity: int
        The verbosity level, 0 = silent, 1 = progress bars and summary.

    :group: input.states.create

    """
    engine = get_engine()
    out_dir = Path(out_dir).expanduser()
    grb_dir = out_dir / "grb"
    nc0_dir = out_dir / "nc"
    nc_dir = nc0_dir / region
    grb_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)
    levels = list(range(69, 75)) if levels is None else levels

    var2ncvar = {
        FV.U: "u",
        FV.V: "v",
        FV.TKE: "tke",
        FV.p: "pres",
        FV.T: "t",
    }

    static_data = StaticData()
    if region == "northsea":
        path_grid_select = static_data.get_file_path(
            STATES, "target_grid_icon_eu_R03B08_nordsee.txt"
        )
    elif region == "baltic":
        path_grid_select = static_data.get_file_path(
            STATES, "target_grid_icon_eu_R03B08_ostsee.txt"
        )
    else:
        raise ValueError(f"Unknown region: {region}, choose 'northsea' or 'baltic'.")
    path_grid_weights = nc0_dir / f"icon_weights_{region}.nc"
    path_icon_grid = nc0_dir / "icon_grid_0027_R03B08_N02.nc"

    def _ymv(vrs=None):
        """Helper to iterate over year/month/var"""
        y, m = min_year, min_month
        while (y < max_year) or (y == max_year and m <= max_month):
            if vrs is not None:
                for v in vrs:
                    yield y, m, v
            else:
                yield y, m
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1

    ym = list(_ymv(vrs=None))
    ymv = list(_ymv(vrs=var2ncvar.keys()))

    # download grid file and prepare grid conversion:
    futures = [
        engine.submit(
            _prepare_grid,
            path_grid_select,
            path_icon_grid,
            path_grid_weights,
            url_icon_grid,
            verbosity=verbosity - 1,
        )
    ]

    # download files in parallel:
    futures += [
        engine.submit(
            _download_icon_dream,
            ymv_i,
            base_url,
            grb_dir,
            verbosity=verbosity - 1,
        )
        for ymv_i in ymv
    ]
    if verbosity > 0:
        results = np.array(
            [
                engine.await_result(f)
                for f in tqdm(futures, desc="Downloading ICON-DREAM files")
            ]
        )
    else:
        results = np.array([engine.await_result(f) for f in futures])

    failed = np.sum(results == -1)
    if verbosity > 0:
        print(
            f"Downloaded {np.sum(results == 1)} files, "
            f"{failed} failed, "
            f"{np.sum(results == 0)} already present."
        )

    if failed > 0:
        if verbosity > 0:
            print("Some downloads failed. Please retry.")
        return
    elif verbosity > 0:
        print(f"All grb files present in {grb_dir}.")

    # process files in parallel:
    futures = [
        engine.submit(
            _process,
            region,
            year,
            month,
            grb_dir,
            nc_dir,
            var2ncvar,
            levels,
            path_grid_select,
            path_grid_weights,
            verbosity=verbosity - 1,
        )
        for year, month in ym
    ]
    results = np.array(
        [
            engine.await_result(f)
            for f in tqdm(
                futures, desc=f"Processing {len(ymv)} GRB files for {len(ym)} months"
            )
        ]
    )
    failed = np.sum(results == -1)
    if verbosity > 0:
        print(
            f"Processed {np.sum(results == 1)} files, "
            f"{failed} failed, "
            f"{np.sum(results == 0)} already present."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dir",
        help="Output directory to save downloaded and processed files",
        type=str,
    )
    parser.add_argument(
        "region",
        help="The region, either 'northsea' or 'baltic'",
        type=str,
        choices=["northsea", "baltic"],
    )
    parser.add_argument(
        "min_year",
        help="Minimal year (inclusive)",
        type=int,
    )
    parser.add_argument(
        "min_month",
        help="Minimal month (inclusive)",
        type=int,
    )
    parser.add_argument(
        "max_year",
        help="Maximal year (inclusive)",
        type=int,
    )
    parser.add_argument(
        "max_month",
        help="Maximal month (inclusive)",
        type=int,
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
        "-v",
        "--verbosity",
        help="The verbosity level, 0 = silent",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    with Engine.new(args.engine, n_procs=args.n_cpus):
        return iconDream2foxes(
            out_dir=args.out_dir,
            region=args.region,
            min_year=args.min_year,
            min_month=args.min_month,
            max_year=args.max_year,
            max_month=args.max_month,
            verbosity=args.verbosity,
        )


if __name__ == "__main__":
    main()
