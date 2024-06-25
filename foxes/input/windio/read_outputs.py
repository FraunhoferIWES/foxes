import numpy as np
from pathlib import Path

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC

def _read_turbine_outputs(wio_outs, odir, out_dicts, verbosity):
    """ Reads the turbine outputs """
    if "turbine_outputs" in wio_outs:
        turbine_outputs = Dict(wio_outs["turbine_outputs"], name="turbine_outputs")
        turbine_nc_filename = turbine_outputs.pop("turbine_nc_filename", "turbine_outputs.nc")
        output_variables = turbine_outputs["output_variables"]
        if verbosity > 1:
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
                        file_path=odir/turbine_nc_filename,
                        verbosity=verbosity,
                    )
                }, name = "turbine_outputs"))

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

    :group: input.windio

    """
    out_dicts = []
    odir = Path(wio_outs.pop("output_folder", "results"))
    odir.mkdir(exist_ok=True, parents=True)
    if verbosity > 1:
        print("  Reading outputs")
        print("    Output folder:", odir)
        print("    Contents:", [k for k in wio_outs.keys()])

    # read turbine_outputs:
    _read_turbine_outputs(wio_outs, odir, out_dicts, verbosity)

        
            
    # read flow field:
    if "flow_field" in wio_outs:
        flow_field = Dict(wio_outs["flow_field"], name="flow_field")
        flow_nc_filename = flow_field.pop("flow_nc_filename", "flow_field.nc")
        if verbosity > 1:
            print("      Reading flow_field")
            print("        File name:", flow_nc_filename)
            print("        Contents:", [k for k in flow_field.keys()])

    # read power table:
    if "power_table" in wio_outs:
        power_table = Dict(wio_outs["power_table"], name="power_table")
        power_table_nc_filename = flow_field.pop("flow_nc_filename", "power_table.nc")
        if verbosity > 1:
            print("  Reading power_table")
            print("    File name:", power_table_nc_filename)
            print("    Contents:", [k for k in power_table.keys()])

    return out_dicts
