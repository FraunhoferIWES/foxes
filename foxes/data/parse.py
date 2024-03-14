from pathlib import Path
import warnings


def parse_Pct_file_name(file_name):
    """
    Parse basic turbine data from file name

    Expected format: `[name]-[...]MW-D[...]-H[...].csv`

    Parameters
    ----------
    file_name: str or pathlib.Path
        Path to the file

    Returns
    -------
    parsed_data: dict
        dict with data parsed from file name

    :group: data

    """
    sname = Path(file_name).stem
    pars = {"name": sname.split(".")[0]}

    i = sname.find(".")
    if i >= 0:
        if "-" in sname[i:]:
            warnings.warn(
                f"Illegal use of '.' in '{sname}', please replace by 'd' for float value dots. Parsing stopped."
            )
            return pars

    if "-" in sname and "_" in sname:
        warnings.warn(
            f"Illegal file name '{file_name}': Contains both '-' and '_'. Parsing stopped."
        )
        return pars

    if "-" in sname:
        pieces = sname.split("-")
    elif "_" in sname:
        pieces = sname.split("_")

    pars["name"] = pieces[0]
    pieces = pieces[1:]
    for p in pieces:
        if p[-1] == "W":
            if p[-2] == "k":
                pars["P_nominal"] = float(p[:-2].replace("d", "."))
            elif p[-2] == "M":
                pars["P_nominal"] = 1.0e3 * float(p[:-2].replace("d", "."))
            elif p[-2] == "G":
                pars["P_nominal"] = 1.0e6 * float(p[:-2].replace("d", "."))
            else:
                pars["P_nominal"] = 1.0e-3 * float(p[:-1].replace("d", "."))
            pars["name"] += "-" + p

        elif p[0] == "D":
            pars["D"] = float(p[1:].replace("d", "."))
        elif p[0] == "H":
            pars["H"] = float(p[1:].replace("d", "."))
        else:
            pars["name"] += "-" + p

    return pars


def parse_Pct_two_files(file_name_A, file_name_B):
    """
    Parse basic turbine data from file names

    Expected format: `[name]-[...]MW-D[...]-H[...].csv`

    Parameters
    ----------
    file_name_A: str or pathlib.Path
        Path to the first file
    file_name_B: str or pathlib.Path
        Path to the second file

    Returns
    -------
    parsed_data: dict
        dict with data parsed from file name

    :group: data

    """
    pars_A = parse_Pct_file_name(file_name_A)
    pars_B = parse_Pct_file_name(file_name_B)
    name = pars_A["name"]
    name_ct = pars_B["name"]
    i = 0
    while len(name) > i and len(name_ct) > i and name[i] == name_ct[i]:
        i += 1
    if i > 0 and name[i - 1] == "-":
        i -= 1
    if i < 1:
        raise ValueError(
            f"Turbine type name not deducible. From file A: '{name}', from file B: '{name_ct}'"
        )
    pars_A["name"] = name[:i]
    pars_B["name"] = name[:i]
    if pars_A != pars_B:
        raise ValueError(
            f"Data parsing from file names failed. File '{file_name_A}' gave '{pars_A}', file '{file_name_B}' gave '{pars_B}'"
        )

    return pars_A
