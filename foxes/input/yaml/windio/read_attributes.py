from foxes.utils import Dict
from foxes.core import WakeModel, WakeFrame
import foxes.variables as FV

from .read_outputs import read_outputs


def _read_wind_deficit(
    wake_model_key,
    wind_deficit,
    superposition,
    induction,
    algo_dict,
    mbook,
    verbosity,
):
    """Reads the wind deficit wake model"""

    wind_def_map = Dict(
        {
            "Jensen": "JensenWake",
            "Bastankhah2014": "Bastankhah2014",
            "Bastankhah2016": "Bastankhah2016",
            "TurbOPark": "TurbOPark",
        },
        name="wind_def_map",
    )

    ws_sup_dict = Dict(
        {
            "Linear": "ws_linear",
            "Quadratic": "ws_quadratic",
            "Product": "ws_product",
        },
        name="ws_sup_dict",
    )
    ws_sup_amb_dict = Dict(
        {
            "Linear": "ws_linear_amb",
            "Quadratic": "ws_quadratic_amb",
            "Product": "ws_product",
        },
        name="ws_sup_dict",
    )

    wname = wind_deficit.pop("name")
    eff_ws = wind_deficit.pop("use_effective_ws", True)
    if verbosity > 2:
        print("    Reading", wake_model_key)
        print("      Name    :", wname)
        print("      Eff ws  :", eff_ws)
        print("      Contents:", [k for k in wind_deficit.keys()])
    wind_def_dict = Dict(wmodel_type=wind_def_map[wname], induction=induction)
    kcoef = Dict(wind_deficit["wake_expansion_coefficient"], name="kcoef")
    ka = kcoef["k_a"]
    kb = kcoef.get("k_b", 0.0)
    amb_ti = kcoef.get("free_stream_ti", False)
    if ka is None or ka == 0.0:
        wind_def_dict["k"] = kb
        if verbosity > 2:
            print("        Using k =", kb)
    else:
        ti_var = FV.AMB_TI if amb_ti else FV.TI
        if verbosity > 2:
            print(f"      Using k = {ka} * {ti_var} + {kb}")
        wind_def_dict["k"] = None
        wind_def_dict["ka"] = ka
        wind_def_dict["kb"] = kb
        wind_def_dict["ti_var"] = ti_var
    if "ceps" in wind_deficit:
        sbf = wind_deficit["ceps"]
        if verbosity > 2:
            print(f"      Using sbeta_factor = {sbf}")
        wind_def_dict["sbeta_factor"] = sbf
    supd = ws_sup_dict if eff_ws else ws_sup_amb_dict
    wind_def_dict["superposition"] = supd[superposition["ws_superposition"]]

    mbook.wake_models[wname] = WakeModel.new(**wind_def_dict)
    if verbosity > 2:
        print(f"      Created wake model '{wname}':")
        print("       ", mbook.wake_models[wname])
    algo_dict["wake_models"].append(wname)

    return ka, kb, amb_ti


def _read_turbulence(
    turbulence_model,
    superposition,
    induction,
    algo_dict,
    mbook,
    ka,
    kb,
    amb_ti,
    verbosity,
):
    """Reads the ti wake model"""

    twake_def_map = Dict(
        {
            "CrespoHernandez": "CrespoHernandezTIWake",
            "IEC-TI-2019": "IECTIWake",
        },
        name="twake_def_map",
    )

    ti_sup_dict = Dict(
        {
            "Linear": "ti_linear",
            "Quadratic": "ti_quadratic",
        },
        name="ti_sup_dict",
    )

    wname = turbulence_model.pop("name")
    if verbosity > 2:
        print("    Reading turbulence_model")
        print("      Name:", wname)
        print("      Contents:", [k for k in turbulence_model.keys()])
    if wname not in ["None", "none"]:
        tiwake_dict = dict(wmodel_type=twake_def_map[wname], induction=induction)
        if wname == "IEC-TI-2019":
            tiwake_dict["opening_angle"] = None
            tiwake_dict["iec_type"] = "2019"
        if "wake_expansion_coefficient" in turbulence_model:
            kcoef = Dict(turbulence_model["wake_expansion_coefficient"], name="kcoef")
            ka = kcoef["k_a"]
            kb = kcoef.get("k_b", 0.0)
            amb_ti = kcoef.get("free_stream_ti", False)
        if ka is None or ka == 0.0:
            tiwake_dict["k"] = kb
            if verbosity > 2:
                print("        Using k =", kb)
        else:
            ti_var = FV.AMB_TI if amb_ti else FV.TI
            if verbosity > 2:
                print(f"      Using k = {ka} * {ti_var} + {kb}")
            tiwake_dict["k"] = None
            tiwake_dict["ka"] = ka
            tiwake_dict["kb"] = kb
            tiwake_dict["ti_var"] = ti_var
        tiwake_dict["superposition"] = ti_sup_dict[superposition["ti_superposition"]]

        mbook.wake_models[wname] = WakeModel.new(**tiwake_dict)
        if verbosity > 2:
            print(f"      Created wake model '{wname}':")
            print("       ", mbook.wake_models[wname])
        algo_dict["wake_models"].append(wname)


def _read_blockage(blockage_model, induction, algo_dict, mbook, verbosity):
    """Reads the blockage model"""
    indc_def_map = Dict(
        {
            "RankineHalfBody": "RankineHalfBody",
            "Rathmann": "Rathmann",
            "SelfSimilarityDeficit": "SelfSimilar",
            "SelfSimilarityDeficit2020": "SelfSimilar2020",
        },
        name="twake_def_map",
    )

    wname = blockage_model.pop("name")
    if verbosity > 2:
        print("    Reading blockage_model")
        print("      Name:", wname)
        print("      Contents:", [k for k in blockage_model.keys()])
    if wname not in ["None", "none"]:
        indc_dict = Dict(wmodel_type=indc_def_map[wname], induction=induction)
        mbook.wake_models[wname] = WakeModel.new(**indc_dict)
        if verbosity > 2:
            print(f"      Created wake model '{wname}':")
            print("       ", mbook.wake_models[wname])
        algo_dict["wake_models"].append(wname)
        algo_dict["algo_type"] = "Iterative"


def _read_rotor_averaging(rotor_averaging, algo_dict, verbosity):
    """Reads the rotor averaging"""
    if verbosity > 2:
        print("    Reading rotor_averaging")
        print("      Contents:", [k for k in rotor_averaging.keys()])
    grid = rotor_averaging["grid"]
    nx = rotor_averaging["n_x_grid_points"]
    ny = rotor_averaging["n_y_grid_points"]
    if nx != ny:
        raise NotImplementedError(
            f"Grid '{grid}': Only nx=ny supported, got nx={nx}, ny={ny}"
        )
    background_averaging = rotor_averaging["background_averaging"]
    wake_averaging = rotor_averaging["wake_averaging"]
    wse_P = rotor_averaging["wind_speed_exponent_for_power"]
    wse_ct = rotor_averaging["wind_speed_exponent_for_ct"]
    if verbosity > 2:
        print("        grid                :", grid)
        print("        background_averaging:", background_averaging)
        print("        wake_averaging      :", wake_averaging)
        print("        ws exponent power   :", wse_P)
        print("        ws exponent ct      :", wse_ct)

    if background_averaging in ["center", "centre"]:
        algo_dict["rotor_model"] = "centre"
    elif background_averaging in ["none", "None", None]:
        algo_dict["rotor_model"] = None
    elif background_averaging == "grid":
        algo_dict["rotor_model"] = f"grid{nx*ny}"
    else:
        algo_dict["rotor_model"] = background_averaging

    if wake_averaging in ["centre", "center"]:
        algo_dict["partial_wakes"] = "centre"
    elif wake_averaging in ["none", "None", "auto", None]:
        algo_dict["partial_wakes"] = None
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

    if verbosity > 2:
        print("        --> rotor_model     :", algo_dict["rotor_model"])
        print("        --> partial_wakes   :", algo_dict["partial_wakes"])


def _read_deflection(deflection, induction, algo_dict, mbook, verbosity):
    """Reads deflection model"""
    defl_def_map = Dict(
        {
            "None": "RotorWD",
            "Batankhah2016": "YawedWakes",
        },
        name="defl_def_map",
    )

    wname = deflection.pop("name")
    if verbosity > 2:
        print("    Reading deflection_model")
        print("      Name:", wname)
        print("      Contents:", [k for k in deflection.keys()])
    indc_dict = Dict(wframe_type=defl_def_map[wname])
    try:
        mbook.wake_frames[wname] = WakeFrame.new(**indc_dict, induction=induction)
    except TypeError:
        mbook.wake_frames[wname] = WakeFrame.new(**indc_dict)
    if verbosity > 2:
        print(f"      Created wake frame '{wname}':")
        print("       ", mbook.wake_frames[wname])
    algo_dict["wake_frame"] = wname


def _read_analysis(wio_ana, idict, mbook, verbosity):
    """Reads the windio analyses"""
    if verbosity > 2:
        print("    Reading analysis")
        print("      Contents:", [k for k in wio_ana.keys()])

    # superposition:
    superposition = Dict(
        wio_ana["superposition_model"], name=wio_ana.name + ".superposition_model"
    )
    if verbosity > 2:
        print("    Reading superposition_model")
        print("      Contents:", [k for k in superposition.keys()])

    # axial induction model:
    imap = Dict(
        {
            "1D": "Betz",
            "Madsen": "Madsen",
        },
        name="induction mapping",
    )
    induction = imap[wio_ana.get("axial_induction_model", "1D")]
    if verbosity > 2:
        print("    axial induction model:", induction)

    # wind deficit model:
    wake_model_key = (
        "wind_deficit_model" if "wind_deficit_model" in wio_ana else "wake_model"
    )
    algo_dict = idict["algorithm"]
    wind_deficit = Dict(
        wio_ana[wake_model_key], name=wio_ana.name + "." + wake_model_key
    )
    ka, kb, amb_ti = _read_wind_deficit(
        wake_model_key,
        wind_deficit,
        superposition,
        induction,
        algo_dict,
        mbook,
        verbosity,
    )

    # turbulence model:
    if "turbulence_model" in wio_ana:
        turbulence_model = Dict(wio_ana["turbulence_model"], name="turbulence_model")
        _read_turbulence(
            turbulence_model,
            superposition,
            induction,
            algo_dict,
            mbook,
            ka,
            kb,
            amb_ti,
            verbosity,
        )
    elif verbosity > 0:
        print("turbulence_model not found, not using a TI wake model")

    # blockage model:
    if "blockage_model" in wio_ana:
        blockage_model = Dict(wio_ana["blockage_model"], name="blockage_model")
        _read_blockage(blockage_model, induction, algo_dict, mbook, verbosity)
    elif verbosity > 0:
        print("blockage_model not found, not using a turbine induction model")

    # rotor_averaging:
    if "rotor_averaging" in wio_ana:
        rotor_averaging = Dict(wio_ana["rotor_averaging"], name="rotor_averaging")
        _read_rotor_averaging(rotor_averaging, algo_dict, verbosity)
    elif verbosity > 0:
        print("rotor_averaging not found, using default settings")

    # deflection:
    if "deflection_model" in wio_ana:
        deflection = Dict(wio_ana["deflection_model"], name="deflection_model")
        _read_deflection(deflection, induction, algo_dict, mbook, verbosity)
    elif verbosity > 0:
        print("deflection_model not found, using default settings")


def read_attributes(wio, idict, mbook, verbosity=1):
    """
    Reads the attributes part of windio

    Parameters
    ----------
    wio: foxes.utils.Dict
        The windio data
    idict: foxes.utils.Dict
        The foxes input data dictionary
    mbook: foxes.models.ModelBook
        The model book
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    odir: pathlib.Path
        The output directory

    :group: input.yaml.windio

    """
    wio_attrs = Dict(wio["attributes"], name=wio.name + ".attributes")
    if verbosity > 1:
        print("Reading attributes")
        print("  Contents:", [k for k in wio_attrs.keys()])

    # read flow model:
    if "flow_model" in wio_attrs:
        flow_model = Dict(wio_attrs["flow_model"], name="flow_model")
        fmname = flow_model.pop("name")
        if verbosity > 2:
            print("    Reading flow_model")
            print("      Name:", fmname)
            print("      Contents:", [k for k in flow_model.keys()])
        if fmname != "foxes":
            print(f"Running flow model 'foxes', overruling original choice '{fmname}'")

    # read analysis:
    wio_ana = Dict(wio_attrs["analysis"], name=wio_attrs.name + ".analysis")
    _read_analysis(wio_ana, idict, mbook, verbosity)

    # outputs:
    odict = idict["outputs"]
    odir = "."
    if "outputs" in wio_attrs:
        outputs = Dict(wio_attrs["outputs"], name=wio_attrs.name + ".outputs")
        odir = read_outputs(outputs, odict, verbosity=verbosity)

    return odir
