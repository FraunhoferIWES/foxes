from foxes.data import FARM, StaticData
from foxes.config import get_input_path
from typing import Any
from foxes.core import WindFarm

from .from_json import add_from_json
from .from_csv import add_from_csv


def add_from_file(
    farm: WindFarm,
    file_path: str,
    *args: Any,
    verbosity: int = 1,
    dbook=None,
    **kwargs: Any,
) -> None:
    """
    Add turbines from file.

    The method is inferred according to the file suffix.

    Parameters
    ----------
    farm
        The wind farm
    file_path
        Path to the file
    verbosity
    args
        Parameters forwarded to the method
    verbosity
        The verbosity level, 0 = silent
    dbook
        The data book, or None for default
    kwargs
        Parameters forwarded to the method

    :group: input.farm_layout

    """

    fpath = get_input_path(file_path)
    dbook = StaticData() if dbook is None else dbook

    if not fpath.is_file():
        if verbosity:
            print(f"Reading static data '{fpath.name}' from context '{FARM}'")
        file_path = dbook.get_file_path(FARM, fpath.name, check_raw=False)

    if fpath.suffix == ".json":
        add_from_json(farm, file_path, *args, **kwargs)
    elif (
        fpath.suffix == ".csv"
        or (len(file_path) > 7 and file_path[-7:] == ".csv.gz")
        or (len(file_path) > 8 and file_path[-8:] == ".csv.bz2")
        or (len(file_path) > 8 and file_path[-8:] == ".csv.zip")
    ):
        ckwargs = {**kwargs, "verbosity": verbosity}
        add_from_csv(farm, file_path, *args, **ckwargs)
    else:
        raise KeyError(
            f"Unsupported file suffix: '{file_path}'. Please provide any of: json, csv, csv.gz, csv.bz2, csv.zip"
        )
