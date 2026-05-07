import numpy as np
from pandas import read_csv

from foxes.core import Turbine
from foxes.config import get_input_path
from foxes.models.turbine_types import TBLFile


def add_from_wrf(
    farm,
    directory,
    mbook,
    txt_file="windturbines.txt",
    tbl_name_fun=lambda i: f"wind-turbine-{i}.tbl",
    csv_file=None,
    csv_col_windfarm="wind_farm",
    csv_col_cluster=None,
    rho=1.225,
    verbosity=1,
    **turbine_parameters,
):
    """
    Add turbines to wind farm from a folder with tbl files
    and a txt file containing lat, lon, tbl_index

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    directory: str
        The directory containing the tbl files and the txt file
    mbook: foxes.ModelBook, optional
        The model book, only needed if tbl_dir is specified
    txt_file: str
        The txt file name
    csv_file: str, optional
        An optional csv file containing additional turbine parameters.
    csv_col_windfarm: str, optional
        The column name in the CSV file for the wind farm identifier
    csv_col_cluster: str, optional
        The column name in the CSV file for the cluster identifier
    rho: float
        The air density for the turbine types, if tbl_dir is given
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    Examples
    --------
    TXT file format: lon, lat, tbl_index

    >>>    57.230095 -1.974240 1
    >>>    57.223583 -2.012743 1
    >>>    57.228312 -2.002316 1
    >>>    57.233467 -1.989579 2
    >>>    ...

    :group: input.farm_layout

    """
    assert farm.data_is_lonlat, "Require `input_is_lonlat=True` in WindFarm constructor"
    if verbosity > 0:
        print("Reading directory", directory)
    directory = get_input_path(directory)
    txt_path = directory / txt_file
    if not txt_path.exists():
        raise FileNotFoundError(f"File {txt_path} not found")
    data = np.genfromtxt(txt_path)
    ttypes = {}
    for i in np.unique(data[:, 2]).astype(int):
        tbl_path = directory / tbl_name_fun(i)
        ttypes[i] = tbl_path.stem
        if not tbl_path.exists():
            raise FileNotFoundError(f"File {tbl_path} not found")
        if verbosity > 1:
            print(f"Creating turbine type: {ttypes[i]}")
        mbook.turbine_types[ttypes[i]] = TBLFile(tbl_path, rho=rho)

    if csv_file is not None:
        csv_path = directory / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"File {csv_path} not found")
        if verbosity > 0:
            print("Reading file", csv_path)
        df = read_csv(csv_path)
        assert len(df.index) == len(data), (
            "CSV file must have the same number of rows as the txt file"
        )
        if csv_col_windfarm is not None:
            wfrm = df[csv_col_windfarm].values
        else:
            wfrm = [turbine_parameters.pop("wind_farm_name", None)] * len(df.index)
        if csv_col_cluster is not None:
            clstr = df[csv_col_cluster].values
        else:
            clstr = [turbine_parameters.pop("cluster_name", None)] * len(df.index)
    else:
        wfrm = [turbine_parameters.pop("wind_farm_name", None)] * len(data)
        clstr = [turbine_parameters.pop("cluster_name", None)] * len(data)

    tmodels = turbine_parameters.pop("turbine_models", [])
    for i, row in enumerate(data):
        ttype = ttypes[int(row[-1])]
        farm.add_turbine(
            Turbine(
                index=i,
                xy=np.flip(row[:2]),
                H=None,
                D=None,
                turbine_models=[ttype] + tmodels,
                wind_farm_name=wfrm[i],
                cluster_name=clstr[i],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
    farm.lock(verbosity=verbosity)
