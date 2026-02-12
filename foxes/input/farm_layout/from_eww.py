import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from os import remove
from zipfile import ZipFile

from foxes.core import Turbine, get_engine, Engine
from foxes.config import get_input_path
from foxes.models.turbine_types import PCtFile
from foxes.utils import download_file


def add_from_eww(
    farm,
    data_source,
    filter={},
    mbook=None,
    csv_dir=None,
    rho=1.225,
    verbosity=1,
    pct_pars={},
    **turbine_parameters,
):
    """
    Add turbines to wind farm via EuroWindWakes database csv input file.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    data_source: str or pandas.DataFrame
        The input csv file or data source
    filter: list of tuple, optional
        A list of filters to apply to the dataframe,
        e.g. ("wind_farm": ["Farm1", "Farm2"]),
        or ("latitude", ">=54.1"). For range filtering, use strings
        that start with ">=", "<=", ">", "<". For exact matches, use single values.
        For multiple matches, use lists, tuples or sets.
    mbook: foxes.ModelBook, optional
        The model book, only needed if csv_dir is specified
    csv_dir: str, optional
        The csv file directory, containing turbine type data files
    rho: float
        The air density for the turbine types, if csv_dir is given
    verbosity: int
        The verbosity level, 0 = silent
    pct_pars: dict
        Additional parameters for the PCtFile constructor
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    Examples
    --------
    Data source format:

    >>>    ,wind_farm,oem_manufacturer,latitude,longitude,country,rated_power,rotor_diameter,hub_height,turbine_type,commissioning_date
    >>>    0,Aberdeen Offshore Wind Farm,Vestas,57.230095,-1.9742404,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    >>>    1,Aberdeen Offshore Wind Farm,Vestas,57.2235827,-2.0127432,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    >>>    2,Aberdeen Offshore Wind Farm,Vestas,57.2169301,-2.0055697,United Kingdom,8.4,164.0,108.5,V164-8.4 MW,2018-09
    >>>    ...

    :group: input.farm_layout

    """
    assert farm.data_is_lonlat, "Require `input_is_lonlat=True` in WindFarm constructor"

    if isinstance(data_source, pd.DataFrame):
        data = data_source
    else:
        if verbosity:
            print("Reading file", data_source)
        pth = get_input_path(data_source)
        data = pd.read_csv(pth, index_col=0, parse_dates=["commissioning_date"])

    comm_dates = data["commissioning_date"].str.strip()
    comm_dates = pd.to_datetime(comm_dates)
    data["comm_year"] = comm_dates.dt.year
    data["comm_month"] = comm_dates.dt.month
    data["comm_day"] = comm_dates.dt.day

    ttypes = data["turbine_type"].unique()
    if csv_dir is not None:
        if mbook is None:
            raise ValueError("Model book must be given if csv_dir is specified")
        csv_dir = get_input_path(csv_dir)
        csv_files = sorted(list(csv_dir.glob("*.csv")))
        csv_names = [f.stem for f in csv_files]
        if verbosity > 0:
            print(f"Reading {len(csv_files)} CSV files from {csv_dir}")
        csv_map = []
        ntt = []
        for t in ttypes:
            if t not in csv_names:
                t = t.replace(" ", "_").replace("/", "_")
            if t not in csv_names:
                raise ValueError(f"Turbine type {t} not found in {csv_dir}")
            ntt.append(t)
            csv_map.append(csv_files[csv_names.index(t)])
        ttypes = ntt
        for t in ttypes:
            mbook.turbine_types[t] = None

    else:
        if verbosity > 0:
            print(
                "No csv_dir specified, assuming turbine types correspond to model book names"
            )

    if filter is not None:
        assert isinstance(filter, list), (
            f"filter must be a list, got {type(filter).__name__}"
        )
        for fi, fdata in enumerate(filter):
            assert isinstance(fdata, tuple), (
                f"filter items must be tuples, got {type(fdata).__name__} at position {fi}"
            )
            k, val = fdata
            assert k in data.columns, f"Column {k} not in data, found {data.columns}"
            if isinstance(val, str):
                if val.startswith(">="):
                    data = data[data[k] >= float(val[2:])]
                elif val.startswith("<="):
                    data = data[data[k] <= float(val[2:])]
                elif val.startswith(">"):
                    data = data[data[k] > float(val[1:])]
                elif val.startswith("<"):
                    data = data[data[k] < float(val[1:])]
                if verbosity > 0:
                    print(f"Applying filter {k}{val}, now {len(data.index)} turbines")
            elif isinstance(val, (list, tuple, set)):
                data = data[data[k].isin(val)]
                if verbosity > 0:
                    print(
                        f"Applying filter {k} in {val}, now {len(data.index)} turbines"
                    )
            else:
                data = data[data[k] == val]
                if verbosity > 0:
                    print(f"Applying filter {k}=={val}, now {len(data.index)} turbines")

    if verbosity > 0:
        print(f"Selected {len(data.index)} turbines from the following wind farms:")
        for wf, g in data.groupby("wind_farm"):
            print(f"  {wf} ({g['commissioning_date'].min()}): {len(g.index)} turbines")

    farms = []
    tmodels = turbine_parameters.pop("turbine_models", [])
    for i0, i in enumerate(data.index):
        fname = str(data.loc[i, "wind_farm"]).replace(" ", "_")
        if fname not in farms:
            farms.append(fname)
            j = 0

        ttype = data.loc[i, "turbine_type"]
        if ttype not in ttypes:
            ttype = ttype.replace(" ", "_").replace("/", "_")
        if csv_dir is not None and mbook.turbine_types[ttype] is None:
            j = ttypes.index(ttype)
            if verbosity > 0:
                print(f"Creating turbine type: {ttype} from file {csv_map[j].name}")
            pars = dict(col_P="power")
            pars.update(pct_pars)
            mbook.turbine_types[ttype] = PCtFile(csv_map[j], rho=rho, **pars)

        lonlat = data.loc[i, ["longitude", "latitude"]].to_numpy(np.float64)
        farm.add_turbine(
            Turbine(
                name=f"{fname}_{j}",
                index=i0,
                xy=lonlat,
                H=data.loc[i, "hub_height"],
                D=data.loc[i, "rotor_diameter"],
                turbine_models=[ttype] + tmodels,
                wind_farm_name=fname,
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )

        j += 1

    farm.lock(verbosity=verbosity)


def download_eww(
    out_folder,
    url_database,
    url_power_curves,
    verbosity=1,
):
    """
    Download EuroWindWakes data files in parallel

    Parameters
    ----------
    out_folder: str
        The output folder for the downloaded files
    url_database: str
        The URL of the EuroWindWakes database csv file
    url_power_curves: str
        The URL of the EuroWindWakes power curve zip file
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    fpath_db: Path
        The path to the downloaded database csv file
    fpath_pc: Path
        The path to the unpacked power curves folder

    """
    engine = get_engine()

    if url_database is None:
        url_database = (
            "https://zenodo.org/records/17311571/files/20251218_eww_opendatabase.csv"
        )
    if url_power_curves is None:
        url_power_curves = "https://zenodo.org/records/17311571/files/power_curves.zip"

    odir = Path(out_folder).expanduser()
    odir.mkdir(parents=True, exist_ok=True)
    fpath_db = odir / "eww_opendatabase.csv"
    fpath_pcz = odir / "power_curves.zip"
    fpath_pc = odir / "power_curves"

    futures = []
    if not fpath_db.exists():
        if verbosity > 0:
            print(f"Downloading EuroWindWakes database to {fpath_db}")
        futures.append(
            engine.submit(
                download_file,
                url_database,
                fpath_db,
                verbosity=verbosity,
            )
        )

    need_pc = len(list(fpath_pc.glob("*.csv"))) == 0
    if need_pc:
        if verbosity > 0:
            print(f"Downloading EuroWindWakes power curves to {fpath_pcz}")
        futures.append(
            engine.submit(
                download_file,
                url_power_curves,
                fpath_pcz,
                verbosity=verbosity,
            )
        )

    results = [engine.await_result(f) for f in futures]
    if any(r == -1 for r in results):
        print("Some downloads failed. Please retry.")
        return None, None
    elif verbosity > 0:
        print(f"All files downloaded to {odir}.")

    if need_pc:
        if verbosity > 0:
            print(f"Unpacking power curves to {fpath_pc}")

        with ZipFile(fpath_pcz, "r") as zip_ref:
            zip_ref.extractall(fpath_pc.parent)

        if verbosity > 1:
            print(f"Removing zip file {fpath_pcz}")
        remove(fpath_pcz)

    return fpath_db, fpath_pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dir",
        help="Output directory to save downloaded files",
        type=str,
    )
    parser.add_argument(
        "--url_db",
        help="URL of the EuroWindWakes database csv file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--url_pc",
        help="URL of the EuroWindWakes power curve zip file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="The verbosity level, 0 = silent",
        default=1,
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
    args = parser.parse_args()

    with Engine.new(args.engine, n_procs=args.n_cpus):
        return download_eww(
            out_folder=args.out_dir,
            url_database=args.url_db,
            url_power_curves=args.url_pc,
            verbosity=args.verbosity,
        )


if __name__ == "__main__":
    main()
