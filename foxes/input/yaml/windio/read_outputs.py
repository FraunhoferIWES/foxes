import numpy as np

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC

from .read_fields import foxes2wio


def _read_turbine_outputs(wio_outs, olist, algo, states_isel, verbosity):
    """Reads the turbine outputs request"""
    if "turbine_outputs" in wio_outs and wio_outs["turbine_outputs"].get_item(
        "report", True
    ):
        turbine_outputs = wio_outs["turbine_outputs"]
        turbine_nc_filename = turbine_outputs.pop_item(
            "turbine_nc_filename", "turbine_outputs.nc"
        )
        output_variables = turbine_outputs["output_variables"]
        if verbosity > 2:
            print("      Reading turbine_outputs")
            print("        File name:", turbine_nc_filename)
            print("        output_variables:", output_variables)

        vmap = Dict(
            power=FV.P,
            rotor_effective_velocity=FV.REWS,
            _name="vmap",
        )
        ivmap = {d: k for k, d in vmap.items()}
        ivmap.update(
            {
                FC.STATE: "time",
                FC.TURBINE: "turbine",
            }
        )

        isel = None
        if states_isel is not None and len(states_isel):
            isel = {"time": states_isel}

        olist.append(
            Dict(
                output_type="StateTurbineTable",
                functions=[
                    dict(
                        function="get_dataset",
                        variables=[vmap[v] for v in output_variables],
                        name_map=ivmap,
                        to_file=turbine_nc_filename,
                        round={
                            vw: FV.get_default_digits(vf) for vw, vf in vmap.items()
                        },
                        isel=isel,
                        verbosity=verbosity,
                    )
                ],
                _name=f"outputs.{len(olist)}.StateTurbineTable",
            )
        )


def _read_flow_field(wio_outs, olist, algo, states_isel, verbosity):
    """Reads the flow field request"""
    if "flow_field" in wio_outs and wio_outs["flow_field"].get_item("report", True):
        flow_field = wio_outs["flow_field"]
        flow_nc_filename = flow_field.pop_item("flow_nc_filename", "flow_field.nc")
        output_variables = flow_field.pop_item("output_variables")

        z_planes = flow_field.pop_item("z_planes")
        z_sampling = z_planes["z_sampling"]
        xy_sampling = z_planes["xy_sampling"]

        if verbosity > 2:
            print("      Reading flow_field")
            print("        File name       :", flow_nc_filename)
            print("        output_variables:", output_variables)
            print("        states subset   :", states_isel)
            print("        z_sampling      :", z_sampling)
            print("        xy_sampling     :", xy_sampling)

        vmap = Dict(
            wind_speed=FV.WS,
            wind_direction=FV.WD,
            _name="vmap",
        )

        z_list = []
        if z_sampling == "plane_list":
            z_list = z_planes.pop_item("z_list")
        elif z_sampling == "hub_heights":
            z_list = np.unique(algo.farm.get_hub_heights(algo))
        elif z_sampling == "grid":
            zb = z_planes.pop_item("z_bounds")
            assert len(zb) == 2, f"Expecting two entries for z_bounds, got {zb}"
            zn = z_planes.pop_item("z_number")
            z_list = np.linspace(zb[0], zb[1], zn)
        elif isinstance(z_sampling, (int, float)):
            z_list = np.atleast_1d(z_sampling)
        else:
            raise NotImplementedError(
                f"z_sampling '{z_sampling}' is not supported. Choices: plane_list, hub_heights, grid."
            )
        z_list = np.asarray(z_list)
        if verbosity > 2:
            print("          z_list        :", z_list)

        if xy_sampling == "grid":
            xb = z_planes.pop_item("x_bounds")
            assert len(xb) == 2, f"Expecting two entries for x_bounds, got {xb}"
            yb = z_planes.pop_item("y_bounds")
            assert len(yb) == 2, f"Expecting two entries for y_bounds, got {yb}"
            dx = z_planes.pop_item("dx")
            dy = z_planes.pop_item("dy")
            nx = max(int((xb[1] - xb[0]) / dx), 1) + 1
            if (xb[1] - xb[0]) / (nx - 1) > dx:
                nx += 1
            ny = max(int((yb[1] - yb[0]) / dy), 1) + 1
            if (yb[1] - yb[0]) / (ny - 1) > dy:
                ny += 1
            z_list = np.asarray(z_list)
            if verbosity > 2:
                print("          x_bounds      :", xb)
                print("          y_bounds      :", yb)
                print("          nx, ny        :", (nx, ny))
                print("          true dx       :", (xb[1] - xb[0]) / (nx - 1))
                print("          true dy       :", (yb[1] - yb[0]) / (ny - 1))
            olist.append(
                Dict(
                    output_type="SlicesData",
                    verbosity_delta=3,
                    functions=[
                        dict(
                            function="get_states_data_xy",
                            z_list=z_list,
                            states_isel=states_isel,
                            xmin=xb[0],
                            xmax=xb[1],
                            ymin=yb[0],
                            ymax=yb[1],
                            n_img_points=(nx, ny),
                            variables=[vmap[v] for v in output_variables],
                            to_file=flow_nc_filename,
                            label_map=foxes2wio,
                            verbosity=verbosity,
                        )
                    ],
                    _name=f"outputs.output{len(olist)}.SliceData",
                )
            )
        else:
            raise NotImplementedError(
                f"xy_sampling '{xy_sampling}' is not supported. Choices: 'grid'."
            )


def read_outputs(wio_outs, idict, algo, verbosity=1):
    """
    Reads the windio outputs

    Parameters
    ----------
    wio_outs: foxes.utils.Dict
        The windio output data dict
    idict: foxes.utils.Dict
        The foxes input data dictionary
    algo: foxes.core.Algorithm
        The algorithm
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    odir: pathlib.Path
        The output directory

    :group: input.yaml.windio

    """
    odir = wio_outs.pop_item("output_folder", ".")
    olist = []
    if verbosity > 2:
        print("  Reading model_outputs_specification")
        print("    Output dir:", odir)
        print("    Contents  :", [k for k in wio_outs.keys()])

    # read subset:
    cases_run = Dict(
        wio_outs.pop_item("cases_run", {}), _name=wio_outs.name + ".cases_run"
    )
    if cases_run.pop_item("all_occurences"):
        states_isel = None
    else:
        states_isel = cases_run.pop_item("subset")

    # read turbine_outputs:
    _read_turbine_outputs(wio_outs, olist, algo, states_isel, verbosity)

    # read flow field:
    _read_flow_field(wio_outs, olist, algo, states_isel, verbosity)

    if len(olist):
        idict["outputs"] = olist

    return odir
