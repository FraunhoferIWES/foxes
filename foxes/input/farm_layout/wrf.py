import numpy as np

from foxes.core import Turbine
from foxes.config import get_input_path
from foxes.models.turbine_types import TBLFile

def add_from_wrf(
    farm,
    directory,
    mbook,
    txt_file="windturbines.txt",
    tbl_name_fun=lambda i: f"wind-turbine-{i}.tbl",
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
    tbl_name_fun: function
        A function that takes an integer index and returns the corresponding tbl file name
    rho: float
        The air density for the turbine types, if tbl_dir is given
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    Examples
    --------
    TXT file format: lon, lat, tbl_index

    57.230095 -1.974240 1
    57.223583 -2.012743 1
    57.228312 -2.002316 1
    57.233467 -1.989579 2
    ...

    :group: input.farm_layout

    """
    assert farm.data_is_lonlat, "Require input_is_lonlat = True in WindFarm constructor"
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
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
    farm.lock(verbosity=verbosity)
