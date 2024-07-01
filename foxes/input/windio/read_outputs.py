import numpy as np
from pathlib import Path

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC

def _read_turbine_outputs(wio_outs, odir, out_dicts, verbosity):
    """ Reads the turbine outputs request """
    if "turbine_outputs" in wio_outs and wio_outs["turbine_outputs"].get("report", True):
        turbine_outputs = Dict(wio_outs["turbine_outputs"], name="turbine_outputs")
        turbine_nc_filename = turbine_outputs.pop("turbine_nc_filename", "turbine_outputs.nc")
        output_variables = turbine_outputs["output_variables"]
        if verbosity > 2:
            print("      Reading turbine_outputs")
            print("        File name:", turbine_nc_filename)
            print("        output_variables:", output_variables)
        
        vmap = Dict(
            power=FV.P,
            rotor_effective_velocity=FV.REWS,
        )
        ivmap = {d: k for k, d in vmap.items()}
        ivmap.update({
            FC.STATE: "time", 
            FC.TURBINE: "turbine",
        })
        
        out_dicts.append(Dict({
                    "output_type": "StateTurbineTable",
                    "farm_results": True,
                    "algo": False,
                    "run_func": "get_dataset",
                    "run_kwargs": dict(
                        variables=[vmap[v] for v in output_variables],
                        name_map=ivmap,
                        to_file=odir/turbine_nc_filename,
                        verbosity=verbosity,
                    ),
                    "output_yaml_update": {
                        "power_table": f"include {turbine_nc_filename}",
                    },
                }, name = "turbine_outputs"))

def _read_flow_field(wio_outs, odir, out_dicts, verbosity):
    """ Reads the flow field request """
    if "flow_field" in wio_outs and wio_outs["flow_field"].get("report", True):
        flow_field = Dict(wio_outs["flow_field"], name="flow_field")
        flow_nc_filename = flow_field.pop("flow_nc_filename", "flow_field.nc")
        output_variables = flow_field.pop("output_variables")
        z_planes = Dict(flow_field.pop("z_planes"), name="z_planes")
        z_sampling = z_planes["z_sampling"]
        xy_sampling = z_planes["xy_sampling"]
        if verbosity > 2:
            print("      Reading flow_field")
            print("        File name       :", flow_nc_filename)
            print("        output_variables:", output_variables)
            print("        z_sampling      :", z_sampling)
            print("        xy_sampling     :", xy_sampling)

        vmap = Dict(
            wind_speed=FV.WS,
            wind_direction=FV.WD,
        )
        
        if xy_sampling == "default":
            out_dicts.append(Dict({
                        "output_type": "SliceData",
                        "farm_results": True,
                        "algo": True,
                        "verbosity_delta": 3,
                        "run_func": "get_states_data_xy",
                        "run_kwargs": dict(
                            resolution=30.,
                            variables=[vmap[v] for v in output_variables],
                            z=None if z_sampling == "hub_height" else z_sampling,
                            to_file=odir/flow_nc_filename,
                            verbosity=verbosity,
                        ),
                        "output_yaml_update": {
                            "flow_field": f"include {flow_nc_filename}",
                        },
                    }, name = "flow_field"))
        else:
            raise NotImplementedError(f"xy_sampling '{xy_sampling}' is not supported (yet)")
            
        
def read_outputs(wio_outs, algo_dict, verbosity):
    """
    Reads the windio outputs

    Parameters
    ----------
    wio_outs: dict
        The windio output data
    algo_dict: dict
        The algorithm dictionary
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    out_dicts: list of dict
        The output dictionaries
    odir: pathlib.Path
        Path to the output folder

    :group: input.windio

    """
    out_dicts = []
    odir = Path(wio_outs.pop("output_folder", "results"))
    odir.mkdir(exist_ok=True, parents=True)
    if verbosity > 2:
        print("  Reading outputs")
        print("    Output folder:", odir)
        print("    Contents:", [k for k in wio_outs.keys()])

    # read turbine_outputs:
    _read_turbine_outputs(wio_outs, odir, out_dicts, verbosity)

    # read flow field:
    _read_flow_field(wio_outs, odir, out_dicts, verbosity)

    return out_dicts, odir
