from foxes.utils import Dict
from foxes.models.turbine_models import kTI
import foxes.variables as FV

def _read_analyses(wio_ana, algo_dict, verbosity):
    """ Reads the windio analyses """
    if verbosity > 1:
        print("    Reading analyses")
        print("      Contents:", [k  for k in wio_ana.keys()])

    wind_def_map = Dict({
        "Jensen": "JensenWake", 
        "Bastankhah2014": "Bastankhah2014", 
        "Bastankhah2016": "Bastankhah2016", 
        "TurbOPark": "TurbOPark",
    }, name="wind_def_map")

    # wind deficit model:
    wind_deficit = Dict(wio_ana["wind_deficit_model"], name="wind_deficit_model")
    wname = wind_deficit.pop("name")
    if verbosity > 1:
        print("    Reading wind_deficit_model")
        print("      Name:", wname)
        print("      Contents:", [k  for k in wind_deficit.keys()])
    wind_def_dict = dict(wmodel_type=wind_def_map[wname])
    kcoef = Dict(wind_deficit["wake_expansion_coefficient"], name="kcoef")
    ka = kcoef["k_a"]
    kb = kcoef.get("k_b", 0.)
    if kb == 0.:
        wind_def_dict["k"] = ka
        if verbosity > 1:
            print("        Using k =", ka)
    else:
        wind_def_dict["k"] = None
        amb_ti = kcoef.get("free_stream_ti", False)
        ti_var = FV.AMB_TI if amb_ti else FV.TI
        if verbosity > 1:
            print(f"      Using k = {ka} + {kb} * {ti_var}")
        algo_dict["mbook"].turbine_models["kTI"] = kTI(kb, ka, ti_var=ti_var)
        for t in algo_dict["farm"].turbines:
            t.models.append("kTI")
    if "ceps" in wind_deficit:
        sbf = wind_deficit["ceps"]
        if verbosity > 1:
            print(f"      Using sbeta_factor = {sbf}")
        wind_def_dict["sbeta_factor"] = sbf

def read_attributes(wio_attrs, algo_dict, verbosity):
    """
    Reads the attributes part of windio
    
    Parameters
    ----------
    wio_attrs: dict
        The windio attributes data
    algo_dict: dict
        The algorithm dictionary
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    out_dicts: list of dict
        The output dictionaries
    
    :group: input.windio

    """
    if verbosity > 1:
        print("Reading attributes")
        print("  Contents:", [k  for k in wio_attrs.keys()])

    # read flow model:
    if "flow_model" in wio_attrs:
        flow_model = Dict(wio_attrs["flow_model"], name="flow_model")
        fmname = flow_model.pop("name")
        if verbosity > 1:
            print("    Reading flow_model")
            print("      Name:", fmname)
            print("      Contents:", [k  for k in flow_model.keys()])
        if fmname != "foxes":
            raise ValueError(f"Can only run flow_model 'foxes', got '{fmname}'")

    _read_analyses(Dict(wio_attrs["analyses"], name="analyses"), algo_dict, verbosity)
    
        
    return []
        