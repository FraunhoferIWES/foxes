"""Translate WindIO reporting requests to constructor-defined output models."""

import numpy as np
from typing import Any

from foxes.utils import Dict
from foxes.core import Algorithm
import foxes.variables as FV
import foxes.constants as FC

from .read_fields import foxes2wio


def _read_state_aggregation(parameters: Dict[Any, Any]) -> bool:
    """Consume the shared boolean state-reduction option."""
    aggregate = parameters.pop_item("aggregate_states", False)
    if not isinstance(aggregate, bool):
        raise ValueError("WindIO: aggregate_states must be a boolean")
    return aggregate


def _read_turbine_outputs(
    wio_outs: Dict[Any, Any],
    olist: list[Any],
    algo: Algorithm,
    states_isel: Any,
    verbosity: int,
) -> None:
    """Append a selected, externally named state/turbine NetCDF table."""
    if "turbine_outputs" in wio_outs and wio_outs["turbine_outputs"].get_item(
        "report", True
    ):
        turbine_outputs = wio_outs["turbine_outputs"]
        turbine_nc_filename = turbine_outputs.pop_item(
            "turbine_nc_filename", "turbine_outputs.nc"
        )
        output_variables = turbine_outputs["output_variables"]
        aggregate = turbine_outputs.pop_item("aggregate_turbines", None)
        aggregate_states = _read_state_aggregation(turbine_outputs)
        if aggregate not in (None, "wind_farm", "cluster", "all"):
            raise ValueError(f"WindIO: Invalid aggregate_turbines {aggregate!r}")
        if verbosity > 2:
            print("      Reading turbine_outputs")
            print("        File name:", turbine_nc_filename)
            print("        output_variables:", output_variables)

        vmap: dict[str, str] = Dict(
            power=FV.P,
            rotor_effective_velocity=FV.REWS,
            _name="vmap",
        )
        ivmap = {
            **{d: k for k, d in vmap.items()},
            FC.STATE: "time",
            FC.TURBINE: "turbine",
            FC.FARM: "wind_farm",
            FC.CLUSTER: "cluster",
        }

        olist.append(
            Dict(
                output_type="StateTurbineTable",
                variables=[vmap[v] for v in output_variables],
                name_map=ivmap,
                to_file=turbine_nc_filename,
                write_pars=dict(
                    round={vw: FV.get_default_digits(vf) for vw, vf in vmap.items()},
                    verbosity=verbosity,
                ),
                states_isel=states_isel,
                aggregate_turbines=aggregate,
                aggregate_states=aggregate_states,
                transpose=True,
                _name=f"outputs.{len(olist)}.StateTurbineTable",
            )
        )


def _flow_heights(planes: Dict[Any, Any], algo: Algorithm) -> np.ndarray:
    """Consume requested plane heights while preserving their supplied order."""
    sampling = planes["z_sampling"]
    if sampling == "plane_list":
        heights = planes.pop_item("z_list")
    elif sampling == "hub_heights":
        heights = np.unique(algo.farm.get_hub_heights(algo))
    elif sampling == "grid":
        bounds = planes.pop_item("z_bounds")
        assert len(bounds) == 2, f"Expecting two entries for z_bounds, got {bounds}"
        heights = np.linspace(bounds[0], bounds[1], planes.pop_item("z_number"))
    elif isinstance(sampling, (int, float)):
        heights = np.atleast_1d(sampling)
    else:
        raise NotImplementedError(
            f"z_sampling '{sampling}' is not supported. Choices: plane_list, hub_heights, grid."
        )
    return np.asarray(heights)


def _grid_count(bounds: list[float], spacing: float) -> int:
    """Include both bounds without exceeding the requested grid spacing."""
    count = max(int((bounds[1] - bounds[0]) / spacing), 1) + 1
    return count + int((bounds[1] - bounds[0]) / (count - 1) > spacing)


def _flow_grid(planes: Dict[Any, Any], verbosity: int) -> dict[str, Any]:
    """Consume horizontal grid sampling and return slice-grid parameters."""
    sampling = planes["xy_sampling"]
    if sampling != "grid":
        raise NotImplementedError(
            f"xy_sampling '{sampling}' is not supported. Choices: 'grid'."
        )
    xb, yb = planes.pop_item("x_bounds"), planes.pop_item("y_bounds")
    assert len(xb) == 2, f"Expecting two entries for x_bounds, got {xb}"
    assert len(yb) == 2, f"Expecting two entries for y_bounds, got {yb}"
    if "dx" in planes or "dy" in planes:
        assert "dx" in planes and "dy" in planes, (
            f"Expecting both 'dx' and 'dy' in z_planes, got {list(planes.keys())}"
        )
        nx = _grid_count(xb, planes.pop_item("dx"))
        ny = _grid_count(yb, planes.pop_item("dy"))
    elif "Nx" in planes or "Ny" in planes:
        assert "Nx" in planes and "Ny" in planes, (
            f"Expecting both 'Nx' and 'Ny' in z_planes, got {list(planes.keys())}"
        )
        nx, ny = planes.pop_item("Nx"), planes.pop_item("Ny")
    else:
        raise KeyError(
            f"Expecting either 'dx' and 'dy' or 'Nx' and 'Ny' in z_planes, got {list(planes.keys())}"
        )
    if verbosity > 2:
        print("          x_bounds      :", xb)
        print("          y_bounds      :", yb)
        print("          nx, ny        :", (nx, ny))
        print("          true dx       :", (xb[1] - xb[0]) / (nx - 1))
        print("          true dy       :", (yb[1] - yb[0]) / (ny - 1))
    return dict(xmin=xb[0], xmax=xb[1], ymin=yb[0], ymax=yb[1], n_img_points=(nx, ny))


def _read_flow_field(
    wio_outs: Dict[Any, Any],
    olist: list[Any],
    algo: Algorithm,
    states_isel: Any,
    verbosity: int,
) -> None:
    """Append a constructor-defined slice output for a WindIO flow-field request.

    Parameters
    ----------
    wio_outs
        WindIO output configuration. The enabled ``flow_field`` entry is
        consumed in place while reading file, field and sampling options.
    olist
        FOXES output-definition list, extended in place with one ``SlicesData``
        entry when flow-field reporting is enabled.
    algo
        Algorithm supplying turbine hub heights when ``z_sampling`` requests
        ``hub_heights``.
    states_isel
        Positional state selection forwarded unchanged to the slice output.
    verbosity
        Print parsed sampling details above level two and use this level for
        NetCDF writing.

    Raises
    ------
    KeyError
        If requested fields lack a WindIO mapping, required options are absent,
        or grid sampling supplies neither spacing nor point counts.
    NotImplementedError
        If horizontal sampling is not ``grid`` or the height-sampling mode is
        unsupported.

    Notes
    -----
    Horizontal bounds and heights are in metres. Heights may be explicit planes,
    unique hub heights, an evenly spaced grid or a single numerical height.
    Horizontal counts retain endpoint coverage; ``wind_speed`` and
    ``wind_direction`` map to ``FV.WS`` and ``FV.WD``.
    """
    if "flow_field" in wio_outs and wio_outs["flow_field"].get_item("report", True):
        flow_field = wio_outs["flow_field"]
        flow_nc_filename = flow_field.pop_item("flow_nc_filename", "flow_field.nc")
        output_variables = flow_field.pop_item("output_variables")
        aggregate_states = _read_state_aggregation(flow_field)
        if flow_field.pop_item("aggregate_points", None) is not None:
            raise ValueError(
                "WindIO: Flow fields remain gridded; aggregate_points is unsupported"
            )

        z_planes = flow_field.pop_item("z_planes")

        if verbosity > 2:
            print("      Reading flow_field")
            print("        File name       :", flow_nc_filename)
            print("        output_variables:", output_variables)
            print("        states subset   :", states_isel)
            print("        z_sampling      :", z_planes["z_sampling"])
            print("        xy_sampling     :", z_planes["xy_sampling"])

        vmap: dict[str, str] = Dict(
            wind_speed=FV.WS,
            wind_direction=FV.WD,
            _name="vmap",
        )

        z_list = _flow_heights(z_planes, algo)
        if verbosity > 2:
            print("          z_list        :", z_list)

        grid = _flow_grid(z_planes, verbosity)
        olist.append(
            Dict(
                output_type="SlicesData",
                positions=z_list.tolist(),
                plane="xy",
                states_isel=states_isel,
                aggregate_states=aggregate_states,
                grid_pars=grid,
                variables=[vmap[v] for v in output_variables],
                to_file=flow_nc_filename,
                label_map={**foxes2wio, FC.STATE: "time"},
                write_pars=dict(verbosity=verbosity),
                _name=f"outputs.{len(olist)}.SlicesData",
            )
        )


def _read_state_subset(times: Dict[Any, Any]) -> list[int] | None:
    """Consume time-subset configuration and reject invalid positional selections."""
    if times.get_item("all_occurences"):
        return None
    selected = times.get_item("subset")
    if (
        not isinstance(selected, list)
        or not selected
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError(
            "WindIO: state subset must contain unique nonnegative "
            "integer positions and must not be empty"
        )
    return selected


def read_outputs(
    wio_outs: Dict[Any, Any],
    idict: dict[str, Any],
    algo: Algorithm,
    verbosity: int = 1,
) -> str:
    """Translate WindIO reporting requests into FOXES output definitions.

    Parameters
    ----------
    wio_outs
        WindIO output specification, consumed while reading reporting options.
    idict
        FOXES input dictionary, extended in place with constructor-defined
        ``StateTurbineTable`` and ``SlicesData`` entries for enabled reports.
    algo
        Algorithm supplying farm geometry for hub-height sampling.
    verbosity
        Parsing and NetCDF-writing verbosity; zero is silent.

    Returns
    -------
    odir
        Requested output folder, or ``"."`` when unspecified.

    Raises
    ------
    KeyError
        If requested fields are unknown or required report options are absent.
    ValueError
        If a time subset is empty or contains repeated, negative or noninteger
        positions, an aggregation option is invalid, or flow-field spatial
        aggregation is requested.
    NotImplementedError
        If wind-speed/direction subsets or sampling modes are unsupported.

    Notes
    -----
    Time subsets retain their specified positional order in both output
    families. Turbine fields have dimensions ``(turbine, time)``; flow fields
    have dimensions ``(z, time, y, x)`` with heights in requested order.
    Turbine ``aggregate_turbines`` selects wind-farm/cluster/all grouping and
    either family accepts boolean ``aggregate_states``. Subsets are selected
    before normalized state reduction. Eliminated dimensions are omitted;
    dataset attributes record their original counts. Flow files stay gridded
    and reject ``aggregate_points``.
    """
    odir = wio_outs.pop_item("output_folder", ".")
    olist: list[Any] = []
    if verbosity > 2:
        print("  Reading model_outputs_specification")
        print("    Output dir:", odir)
        print("    Contents  :", [k for k in wio_outs.keys()])

    # read subset:
    run_configuration = wio_outs.pop_item("run_configuration", {})
    states_isel = None
    if "times_run" in run_configuration:
        states_isel = _read_state_subset(run_configuration.pop_item("times_run"))
    elif "wind_speeds_run" in run_configuration:
        wind_speeds_run = run_configuration.get_item("wind_speeds_run")
        directions_run = run_configuration.get_item("directions_run")
        if not wind_speeds_run.get_item("all_values"):
            raise NotImplementedError(
                f"Wind speed and direction subsets are not yet supported, got {wind_speeds_run.name} {wind_speeds_run}"
            )
        if not directions_run.get_item("all_values"):
            raise NotImplementedError(
                f"Wind speed and direction subsets are not yet supported, got {directions_run.name} {directions_run}"
            )

    # read turbine_outputs:
    _read_turbine_outputs(wio_outs, olist, algo, states_isel, verbosity)

    # read flow field:
    _read_flow_field(wio_outs, olist, algo, states_isel, verbosity)

    if len(olist):
        idict["outputs"] = olist

    return odir
