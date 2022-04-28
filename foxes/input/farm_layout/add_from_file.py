from pathlib import Path

from foxes.input.farm_layout.add_from_json import add_from_json
from foxes.input.farm_layout.add_from_csv import add_from_csv

def add_from_file(farm, file_path, *args, **kwargs):

    fpath = Path(file_path)
    if fpath.suffix == ".json":
        add_from_json(farm, file_path, *args, **kwargs)
    elif fpath.suffix == ".csv" \
        or ( len(fpath) > 7 and fpath[-7:] == ".csv.gz" ) \
        or ( len(fpath) > 8 and fpath[-8:] == ".csv.bz2" ) \
        or ( len(fpath) > 8 and fpath[-8:] == ".csv.zip" ):
        add_from_csv(farm, file_path, *args, **kwargs)
    else:
        raise KeyError(f"Unsupported file suffix: '{file_path}'. Please provide any of: json, csv, csv.gz, csv.bz2, csv.zip")
