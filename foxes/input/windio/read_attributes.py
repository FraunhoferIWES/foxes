from foxes.utils import Dict
from foxes.core import WakeModel, RotorModel
import foxes.variables as FV

def _read_analysis(wio_ana, algo_dict, verbosity):
    """ Reads the windio analyses """
    if verbosity > 1:
        print("    Reading analysis")
        print("      Contents:", [k  for k in wio_ana.keys()])

    wind_def_map = Dict({
        "Jensen": "JensenWake", 
        "Bastankhah2014": "Bastankhah2014", 
        "Bastankhah2016": "Bastankhah2016", 
        "TurbOPark": "TurbOPark",
    }, name="wind_def_map")

    twake_def_map = Dict({
        "CrespoHernandez": "CrespoHernandez",
    }, name="twake_def_map")

    ws_sup_dict = Dict({
        "Linear": "ws_linear",
        "Quadratic": "ws_quadratic",
    }, name="ws_sup_dict")

    # read superposition:
    superposition_model = Dict(wio_ana["superposition_model"], name="superposition_model")
    if verbosity > 1:
        print("    Reading superposition_model")
        print("      Contents:", [k  for k in superposition_model.keys()])

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
    if ka is None or ka == 0.:
        wind_def_dict["k"] = kb
        if verbosity > 1:
            print("        Using k =", kb)
    else:
        amb_ti = kcoef.get("free_stream_ti", False)
        ti_var = FV.AMB_TI if amb_ti else FV.TI
        if verbosity > 1:
            print(f"      Using k = {ka} * {ti_var} + {kb}")
        wind_def_dict["k"] = None
        wind_def_dict["ka"] = ka
        wind_def_dict["kb"] = kb
        wind_def_dict["ti_var"] = ti_var
    if "ceps" in wind_deficit:
        sbf = wind_deficit["ceps"]
        if verbosity > 1:
            print(f"      Using sbeta_factor = {sbf}")
        wind_def_dict["sbeta_factor"] = sbf
    wind_def_dict["superposition"] = ws_sup_dict[superposition_model["ws_superposition"]]

    algo_dict["mbook"].wake_models[wname] = WakeModel.new(**wind_def_dict)
    if verbosity > 1:
        print(f"      Created wake model '{wname}':")
        print("       ", algo_dict["mbook"].wake_models[wname])
    algo_dict["wake_models"].append(wname)

    # turbulence model:
    turbulence_model = Dict(wio_ana["turbulence_model"], name="turbulence_model")
    tname = turbulence_model.pop("name")
    if verbosity > 1:
        print("    Reading turbulence_model")
        print("      Name:", tname)
        print("      Contents:", [k  for k in turbulence_model.keys()])
    twake_dict = dict(wmodel_type=twake_def_map[tname])
    kcoef = Dict(wind_deficit["wake_expansion_coefficient"], name="kcoef")
    ka = kcoef["k_a"]
    kb = kcoef.get("k_b", 0.)
    if ka is None or ka == 0.:
        wind_def_dict["k"] = kb
        if verbosity > 1:
            print("        Using k =", kb)
    else:
        amb_ti = kcoef.get("free_stream_ti", False)
        ti_var = FV.AMB_TI if amb_ti else FV.TI
        if verbosity > 1:
            print(f"      Using k = {ka} * {ti_var} + {kb}")
        wind_def_dict["k"] = None
        wind_def_dict["ka"] = ka
        wind_def_dict["kb"] = kb
        wind_def_dict["ti_var"] = ti_var
    if "ceps" in wind_deficit:
        sbf = wind_deficit["ceps"]
        if verbosity > 1:
            print(f"      Using sbeta_factor = {sbf}")
        wind_def_dict["sbeta_factor"] = sbf
    wind_def_dict["superposition"] = ws_sup_dict[superposition_model["ws_superposition"]]
   
    # rotor_averaging:
    rotor_averaging = Dict(wio_ana["rotor_averaging"], name="rotor_averaging")
    if verbosity > 1:
        print("    Reading rotor_averaging")
        print("      Contents:", [k  for k in rotor_averaging.keys()])
    grid = rotor_averaging["grid"]
    nx = rotor_averaging["n_x_grid_points"]
    ny = rotor_averaging["n_y_grid_points"]
    if nx != ny:
        raise NotImplementedError(f"Grid '{grid}': Only nx=ny supported, got nx={nx}, ny={ny}")
    background_averaging = rotor_averaging["background_averaging"]
    wake_averaging = rotor_averaging["wake_averaging"]
    wse_P = rotor_averaging["wind_speed_exponent_for_power"]
    wse_ct = rotor_averaging["wind_speed_exponent_for_ct"]
    if verbosity > 1:
        print("        grid                :", grid)
        print("        background_averaging:", background_averaging)
        print("        wake_averaging      :", wake_averaging)
        print("        ws exponent power   :", wse_P)
        print("        ws exponent ct      :", wse_ct)
    if background_averaging == "center":
        algo_dict["rotor_model"] = "centre"
    elif background_averaging == "grid":
        algo_dict["rotor_model"] = f"grid{nx*ny}"
    else:
        raise KeyError(f"Expecting background_averaging 'center' or 'grid', got '{background_averaging}'")
    if wake_averaging == "centre":
        algo_dict["partial_wakes"] = "centre"
    elif wake_averaging == "grid":
        if background_averaging == "grid":
            algo_dict["partial_wakes"] = "rotor_points"
        else:
            if grid == "grid":
                algo_dict["partial_wakes"] = f"grid{nx*ny}"
            else:
                algo_dict["partial_wakes"] = grid
    else:
        algo_dict["partial_wakes"] = wake_averaging
    if verbosity > 1:
        print("        --> rotor_model     :", algo_dict["rotor_model"])
        print("        --> partial_wakes   :", algo_dict["partial_wakes"])

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
    if verbosity > 0:
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

    _read_analysis(Dict(wio_attrs["analysis"], name="analyses"), algo_dict, verbosity)
    
        
    return []
        