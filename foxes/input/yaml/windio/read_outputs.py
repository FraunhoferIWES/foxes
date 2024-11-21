from pathlib import Path

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC

from .read_fields import foxes2wio


def _read_turbine_outputs(wio_outs, odict, verbosity):
    """Reads the turbine outputs request"""
    if "turbine_outputs" in wio_outs and wio_outs["turbine_outputs"].get(
        "report", True
    ):
        turbine_outputs = Dict(
            wio_outs["turbine_outputs"], name=wio_outs.name + ".turbine_outputs"
        )
        turbine_nc_filename = turbine_outputs.pop(
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
        )
        ivmap = {d: k for k, d in vmap.items()}
        ivmap.update(
            {
                FC.STATE: "time",
                FC.TURBINE: "turbine",
            }
        )

        odict["StateTurbineTable"] = Dict(
            functions=[
                dict(
                    name="get_dataset",
                    variables=[vmap[v] for v in output_variables],
                    name_map=ivmap,
                    to_file=turbine_nc_filename,
                    round={vw: FV.get_default_digits(vf) for vw, vf in vmap.items()},
                    verbosity=verbosity,
                )
            ],
            name=odict.name + ".StateTurbineTable",
        )


def _read_flow_field(wio_outs, odict, verbosity):
    """Reads the flow field request"""
    if "flow_field" in wio_outs and wio_outs["flow_field"].get("report", True):
        flow_field = Dict(wio_outs["flow_field"], name=wio_outs.name + ".flow_field")
        flow_nc_filename = flow_field.pop("flow_nc_filename", "flow_field.nc")
        output_variables = flow_field.pop("output_variables")
        z_planes = Dict(flow_field.pop("z_planes"), name=flow_field.name + ".z_planes")
        z_sampling = z_planes["z_sampling"]
        xy_sampling = z_planes["xy_sampling"]
        cases_run = Dict(
            flow_field.pop("cases_run", {}), name=flow_field.name + ".cases_run"
        )
        states_isel = cases_run.get("subset", None)
        if "all_occurences" in cases_run and cases_run.pop("all_occurences"):
            states_isel = None
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
        )

        if z_sampling in ["hub_height", "default"]:
            z = None
        elif isinstance(z_sampling, (int, float)):
            z = z_sampling
        else:
            raise NotImplementedError(
                f"z_sampling '{z_sampling}' of type '{type(z_sampling).__name__}' is not supported (yet). Please give 'hub_height', 'default' or a float."
            )

        if xy_sampling == "default":
            odict["SliceData"] = Dict(
                verbosity_delta=3,
                functions=[
                    dict(
                        name="get_states_data_xy",
                        states_isel=states_isel,
                        n_img_points=(100, 100),
                        variables=[vmap[v] for v in output_variables],
                        z=z,
                        to_file=flow_nc_filename,
                        label_map=foxes2wio,
                        verbosity=verbosity,
                    )
                ],
                name=odict.name + ".SliceData",
            )
        else:
            raise NotImplementedError(
                f"xy_sampling '{xy_sampling}' is not supported (yet)"
            )


def read_outputs(wio_outs, odict, verbosity=1):
    """
    Reads the windio outputs

    Parameters
    ----------
    wio_outs: foxes.utils.Dict
        The windio output data dict
    odict: foxes.utils.Dict
        The foxes output dictionary
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    odir: pathlib.Path
        The output directory

    :group: input.yaml.windio

    """
    odir = wio_outs.pop("output_folder", ".")
    if verbosity > 2:
        print("  Reading outputs")
        print("    Output dir:", odir)
        print("    Contents  :", [k for k in wio_outs.keys()])

    # read turbine_outputs:
    _read_turbine_outputs(wio_outs, odict, verbosity)

    # read flow field:
    _read_flow_field(wio_outs, odict, verbosity)

    return odir
