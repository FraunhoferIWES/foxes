from pathlib import Path

from foxes.data import get_static_path, FARM
from foxes.input.farm_layout.add_from_json import add_from_json
from foxes.input.farm_layout.add_from_csv import add_from_csv

def add_from_file(farm, file_path, *args, verbosity=1, **kwargs):
    """
    Add turbines from file.

    The method is inferred according to the file suffix.

    Parameters
    ----------
    farm : foxes.WindFarm
        The wind farm
    file_path : str
        Path to the file
    verbosity
    args : tuple, optional
        Parameters forwarded to the method
    verbosity : int
        The verbosity level, 0 = silent
    kwargs : dict, optional
        Parameters forwarded to the method

    """

    fpath = Path(file_path)

    if not fpath.is_file():
        if verbosity:
            print(f"Reading static data '{file_path}' from context '{FARM}'")
        fpath = get_static_path(FARM, file_path)

    if fpath.suffix == ".json":
        add_from_json(farm, fpath, *args, **kwargs)
    elif fpath.suffix == ".csv" \
        or ( len(fpath) > 7 and fpath[-7:] == ".csv.gz" ) \
        or ( len(fpath) > 8 and fpath[-8:] == ".csv.bz2" ) \
        or ( len(fpath) > 8 and fpath[-8:] == ".csv.zip" ):
        add_from_csv(farm, fpath, *args, verbosity=verbosity, **kwargs)
    else:
        raise KeyError(f"Unsupported file suffix: '{fpath}'. Please provide any of: json, csv, csv.gz, csv.bz2, csv.zip")
