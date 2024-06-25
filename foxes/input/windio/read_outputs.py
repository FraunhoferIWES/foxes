import numpy as np
from pathlib import Path

from foxes.utils import Dict
import foxes.constants as FC

from .read_fields import _read_multi_dimensional_coordinate, _read_multi_dimensional_data

def _read_coordinate(wio_data, verbosity):
    """ Reads a coordinate """
    return np.array([wio_data["x"], wio_data["y"]], dtype=FC.DTYPE)

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
        print("    Contents:", [k  for k in wio_outs.keys()])
    
    # read flow field:
    if "flow_field" in wio_outs:
        flow_field = Dict(wio_outs["flow_field"], name="flow_field")
        flow_nc_filename = flow_field.pop("flow_nc_filename", "flow_field.nc")
        if verbosity > 1:
            print("  Reading flow_field")
            print("    File name:", flow_nc_filename)
            print("    Contents:", [k  for k in flow_field.keys()])
    
    # read power table:
    if "power_table" in wio_outs:
        power_table = Dict(wio_outs["power_table"], name="power_table")
        power_table_nc_filename = flow_field.pop("flow_nc_filename", "power_table.nc")
        if verbosity > 1:
            print("  Reading power_table")
            print("    File name:", power_table_nc_filename)
            print("    Contents:", [k  for k in power_table.keys()])
        
        coords = {}
        fields = {}
        dims = {}
        for k, d in power_table.items():
            if _read_multi_dimensional_coordinate(k, d, coords):
                if verbosity > 1:
                    print(f"      Adding coordinate {k}, shape: {coords[k].shape}")
            elif _read_multi_dimensional_data(k, d, fields, dims):
                if verbosity > 1:
                    print(f"      Adding field {k}, dims: {dims[k]}, shape: {fields[k].shape}")
                
                
    quit()
    return out_dicts