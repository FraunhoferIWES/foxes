from importlib import resources

from . import farms
from . import states
from . import power_ct_curves

def _get_pkg(context):
    if context == "farm":
        return farms
    elif context == "states":
        return states
    elif context == "power_ct_curve":
        return power_ct_curves
    else:
        raise KeyError(f"Unknown context '{context}', choices: farm, states, power_ct_curve")

def _get_sfx(context):
    if context == "farm":
        return ".csv"
    elif context == "states":
        return ".csv.gz"
    elif context == "power_ct_curve":
        return ".csv"
    else:
        raise KeyError(f"Unknown context '{context}', choices: farm, states, power_ct_curve")

def get_static_path(context, data_name):
    return resources.path(_get_pkg(context), data_name + _get_sfx(data_name))

def static_contents(context):
    s = _get_sfx(context)
    n = len(s)
    return [c[:-n] for c in resources.contents(_get_pkg(context)) if s in c]

def read_static_file(context, data_name):
    pkg   = _get_pkg(context)
    sfx   = _get_sfx(context)
    fname = data_name + sfx
    try:
        if sfx == ".csv":
            return resources.open_text(pkg, fname)
        else:
            return resources.open_binary(pkg, fname)
    except FileNotFoundError:
        e = f"Could not find static data '{data_name}' for context '{context}'. Available data: {static_contents(context)}"
        raise FileExistsError(e)
