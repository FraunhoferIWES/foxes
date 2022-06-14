from importlib import resources
from pathlib import Path

from . import farms
from . import states
from . import power_ct_curves

FARM     = "farm"
STATES   = "states"
PCTCURVE = "power_ct_curve"

def _get_pkg(context):
    """
    Helper for translating context to package
    """
    if context == FARM:
        return farms
    elif context == STATES:
        return states
    elif context == PCTCURVE:
        return power_ct_curves
    else:
        raise KeyError(f"Unknown context '{context}', choices: {FARM}, {STATES}, {PCTCURVE}")

def _get_sfx(context):
    """
    Helper for translating context to file suffix
    """
    if context == FARM:
        return ".csv"
    elif context == STATES:
        return ".csv.gz"
    elif context == PCTCURVE:
        return ".csv"
    else:
        raise KeyError(f"Unknown context '{context}', choices: {FARM}, {STATES}, {PCTCURVE}")

def static_contents(context):
    """
    Get list of static content for the given context

    Parameters
    ----------
    context : str
        The data context: farm, states, power_ct_curve
    
    Returns
    -------
    contents : list of str
        The available data names

    """
    s = _get_sfx(context)
    n = len(s)
    return [c[:-n] for c in resources.contents(_get_pkg(context)) if s in c]

def get_static_path(context, data_name):
    """
    Get path to static data

    Parameters
    ----------
    context : str
        The data context: farm, states, power_ct_curve
    data_name : str
        The data name (without suffix)
    
    Returns
    -------
    path : pathlib.PosixPath
        Path to the static file

    """
    pkg = _get_pkg(context)
    sfx = _get_sfx(context)
    try:
        with resources.path(pkg, data_name + sfx) as path:
            return path
    except FileNotFoundError:
        raise FileNotFoundError(f"Data '{data_name}' not found in context '{context}'. Available: {static_contents(context)}")

def read_static_file(context, data_name):
    """
    Read a static (non-gz) file

    Parameters
    ----------
    context : str
        The data context: farm, states, power_ct_curve
    data_name : str
        The data name (without suffix)
    
    Returns
    -------
    file: file_object
        The file

    """

    pkg   = _get_pkg(context)
    sfx   = _get_sfx(context)
    fname = data_name + sfx

    if sfx[-3:] == ".gz":
        raise NotImplementedError(f"Cannot run read_static_file on gz type file '{fname}'. Use get_static_path and read manually instead")

    try:
        return resources.open_text(pkg, fname)
    except FileNotFoundError:
        e = f"Could not find static data '{data_name}' for context '{context}'. Available data: {static_contents(context)}"
        raise FileExistsError(e)

def parse_Pct_file_name(file_name):
    """
    Parse file name data

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
    pars  = {"name": sname.split(".")[0]}

    i = sname.find(".")
    if i >= 0:
        if "-" in sname[i:]:
            raise ValueError(f"Illegal use of '.' in '{sname}', please replace by 'd' for float value dots")

    pieces = sname.split("-")[1:]
    for p in pieces:

        if p[-1] == "W":
            if p[-2] == "k":
                pars["P_nominal"] = float(p[:-2])
            elif p[-2] == "M":
                pars["P_nominal"] = 1.e3 * float(p[:-2])
            elif p[-2] == "G":
                pars["P_nominal"] = 1.e6 * float(p[:-2])  
            else:
                pars["P_nominal"] = 1.e-3 * float(p[:-1])

        elif p[0] == "D":
            pars["D"] = float(p[1:].replace("d", "."))
        elif p[0] == "H":
            pars["H"] = float(p[1:].replace("d", "."))
        else:
            raise ValueError(f"Failed to parse piece '{p}' of '{sname}'")

    return pars
