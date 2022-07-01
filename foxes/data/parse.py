from pathlib import Path
import warnings


def parse_Pct_file_name(file_name):
    """
    Parse file name data

    Expected format: `[name]-[...]MW-D[...]-H[...].csv`

    Parameters
    ----------
    file_name : str or pathlib.Path
        Path to the file

    Returns
    -------
    parsed_data : dict
        dict with data parsed from file name

    """
    sname = Path(file_name).stem
    pars = {"name": sname.split(".")[0]}

    i = sname.find(".")
    if i >= 0:
        if "-" in sname[i:]:
            warnings.warn(
                f"Illegal use of '.' in '{sname}', please replace by 'd' for float value dots"
            )
            return pars

    pieces = sname.split("-")[1:]
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

        elif p[0] == "D":
            pars["D"] = float(p[1:].replace("d", "."))
        elif p[0] == "H":
            pars["H"] = float(p[1:].replace("d", "."))
        else:
            warnings.warn(f"Failed to parse piece '{p}' of '{sname}'")

    return pars
