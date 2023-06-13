from pathlib import Path

from foxes.data import FARM, StaticData
from .from_json import add_from_json
from .from_csv import add_from_csv


def add_from_file(farm, file_path, *args, verbosity=1, dbook=None, **kwargs):
    """
    Add turbines from file.

    The method is inferred according to the file suffix.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    file_path: str
        Path to the file
    verbosity
    args: tuple, optional
        Parameters forwarded to the method
    verbosity: int
        The verbosity level, 0 = silent
    dbook: foxes.DataBook, optional
        The data book, or None for default
    kwargs: dict, optional
        Parameters forwarded to the method

    :group: input.farm_layout

    """

    fpath = Path(file_path)
    dbook = StaticData() if dbook is None else dbook

    if not fpath.is_file():
        if verbosity:
            print(f"Reading static data '{fpath.name}' from context '{FARM}'")
        fpath = dbook.get_file_path(FARM, fpath.name, check_raw=False)

    if fpath.suffix == ".json":
        add_from_json(farm, fpath, *args, **kwargs)
    elif (
        fpath.suffix == ".csv"
        or (len(fpath) > 7 and fpath[-7:] == ".csv.gz")
        or (len(fpath) > 8 and fpath[-8:] == ".csv.bz2")
        or (len(fpath) > 8 and fpath[-8:] == ".csv.zip")
    ):
        add_from_csv(farm, fpath, *args, verbosity=verbosity, **kwargs)
    else:
        raise KeyError(
            f"Unsupported file suffix: '{fpath}'. Please provide any of: json, csv, csv.gz, csv.bz2, csv.zip"
        )
