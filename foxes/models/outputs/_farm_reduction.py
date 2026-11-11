"""Compact sufficient statistics for constructor-defined farm evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV

from ._array_reduction import ArrayStats, merge_stats, reduce_array, stats_value


STATE_OPERATIONS = frozenset(
    {
        "reduce_states",
        "reduce_all",
        "calc_states_mean",
        "calc_states_sum",
        "calc_states_std",
        "calc_farm_mean",
        "calc_farm_sum",
        "calc_mean_farm_power",
        "calc_yield",
        "calc_farm_yield",
        "calc_farm_capacity_factor",
        "calc_farm_efficiency",
    }
)
TURBINE_OPERATIONS = frozenset(
    {
        "reduce_turbines",
        "calc_turbine_mean",
        "calc_turbine_sum",
    }
)
METADATA_OPERATIONS = frozenset(
    {"get_capacity", "get_power_units", "calc_farm_capacity"}
)


def reduction_plan(
    operation: str, parameters: dict[str, Any], fields: dict[str, Any]
) -> dict[str, str | None]:
    """Resolve variable reductions for a constructor-defined farm evaluation.

    Parameters
    ----------
    operation
        Evaluation name, including explicit reduction or derived power metrics.
    parameters
        Operation arguments with variable selections and reduction choices.
    fields
        Available variable-to-``(dimensions, values)`` mapping. Dimensions are
        inspected only when an automatic state-reduction plan is requested.

    Returns
    -------
    plan
        Selected variable names mapped to their reduction tokens. ``None``
        retains a static field. Unrequested variables are omitted.

    Raises
    ------
    TypeError
        If turbine reduction has no explicit ``vars_op`` mapping.
    """
    if operation in ("reduce_states", "reduce_turbines"):
        plan = parameters.get("vars_op")
        if plan is not None:
            return dict(plan)
        if operation == "reduce_turbines":
            raise TypeError("reduce_turbines requires 'vars_op'")
        plan = {
            v: ("sum" if v in FV.extensive_state else "weights")
            for v, (dims, _) in fields.items()
            if dims == (FC.STATE, FC.TURBINE)
        }
        plan.update(
            {v: None for v, (dims, _) in fields.items() if dims == (FC.TURBINE,)}
        )
        return plan
    if operation == "reduce_all":
        return {v: parameters["states_op"][v] for v in parameters["turbines_op"]}
    if operation in (
        "calc_mean_farm_power",
        "calc_yield",
        "calc_farm_yield",
        "calc_farm_capacity_factor",
    ):
        return {FV.AMB_P if parameters.get("ambient", False) else FV.P: "weights"}
    if operation == "calc_farm_efficiency":
        return {FV.P: "weights", FV.AMB_P: "weights"}
    variables = parameters["vars"]
    if operation == "calc_states_mean" and isinstance(variables, str):
        variables = [variables]
    choice = "weights"
    if operation.endswith("_sum"):
        choice = "sum"
    elif operation.endswith("_std"):
        choice = "std"
    elif operation in TURBINE_OPERATIONS or not parameters.get("use_weights", True):
        choice = "mean"
    return {v: choice for v in variables}


def required_fields(
    operation: str, parameters: dict[str, Any]
) -> tuple[str, ...] | None:
    """Select the scientific fields required by a configured evaluation.

    Parameters
    ----------
    operation
        Evaluation name.
    parameters
        Operation arguments used to select fields and metadata.

    Returns
    -------
    variables
        Unique field names, including weights/capacity when required. An empty
        tuple denotes metadata-only work; ``None`` collects all available fields.
    """
    if operation in METADATA_OPERATIONS:
        return ()
    if operation in STATE_OPERATIONS | TURBINE_OPERATIONS:
        if operation == "reduce_states" and parameters.get("vars_op") is None:
            return None
        fields = list(reduction_plan(operation, parameters, {}))
        fields.append(FV.WEIGHT)
        if operation == "calc_farm_capacity_factor":
            fields.append(FV.CAP)
        return tuple(dict.fromkeys(fields))
    if operation == "get_data":
        return (parameters["var"], FV.WEIGHT)
    if operation == "gen_stdata":
        return (parameters["variable"],)
    return None


def _validated_field(chunk: dict[str, Any], variable: str) -> np.ndarray:
    """Read a field while rejecting NaNs that carry nonzero state weight.

    Parameters
    ----------
    chunk
        Farm payload with dimensioned fields and optional state weights.
    variable
        Field to validate without modifying its values.

    Returns
    -------
    values
        Original field array. Nonnumeric fields pass through unchanged.

    Raises
    ------
    AssertionError
        If NaNs occur without state weights or at nonzero-weight positions.
    """
    dims, values = chunk["data_vars"][variable]
    try:
        mask = np.isnan(values)
    except TypeError:
        return values
    if not np.any(mask):
        return values
    assert FC.STATE in dims and FV.WEIGHT in chunk["data_vars"], (
        f"Found {np.sum(mask)} nan values for variable '{variable}' of shape {values.shape} without state weights"
    )
    weight_dims, weights = chunk["data_vars"][FV.WEIGHT]
    if weight_dims == (FC.STATE,) and values.ndim == 2:
        weights = weights[:, None]
    invalid = mask & (weights != 0.0)
    assert not np.any(invalid), (
        f"Found {np.sum(invalid)} nan values for variable '{variable}' of shape {values.shape} with nonzero weights"
    )
    return values


def _reduce_field(
    chunk: dict[str, Any], variable: str, choice: str | None, turbine_mode: bool
) -> ArrayStats:
    """Reduce one validated farm field along the selected scientific axis.

    Parameters
    ----------
    chunk
        Farm payload with dimensioned fields and state weights.
    variable
        Field whose sufficient statistics are required.
    choice
        Reduction token from the operation plan.
    turbine_mode
        Eliminate turbines rather than states. Per-state results are finalized
        immediately because distinct state chunks will be concatenated.

    Returns
    -------
    statistics
        Reduced values and only the counts/moments needed for later merging.

    Raises
    ------
    KeyError
        If the reduction is not supported on the selected axis.
    ValueError
        If state-weight dimensions are invalid.
    """
    allowed = (
        ("weights", "weights_sum", "mean", "sum", "min", "max")
        if turbine_mode
        else (None, "weights", "mean", "sum", "min", "max", "std")
    )
    if choice not in allowed:
        raise KeyError(f"Unknown output reduction '{choice}' for variable '{variable}'")
    values = _validated_field(chunk, variable)
    weights = None
    if choice in ("weights", "weights_sum"):
        weight_dims, weights = chunk["data_vars"][FV.WEIGHT]
        if weight_dims == (FC.STATE,):
            weights = weights[:, None]
        elif weight_dims != (FC.STATE, FC.TURBINE):
            raise ValueError(f"Invalid state-weight dimensions {weight_dims}")
    actual = "weights" if choice == "weights_sum" else choice
    stats = reduce_array(values, actual, axis=1 if turbine_mode else 0, weights=weights)
    if turbine_mode and choice == "weights":
        stats["value"] /= values.shape[1]
    return {"value": stats_value(stats, actual)} if turbine_mode else stats


def _reduce_scalar_fields(
    operation: str,
    parameters: dict[str, Any],
    plan: dict[str, str | None],
    stats: dict[str, ArrayStats],
    power_units: np.ndarray | None,
) -> set[str]:
    """Collapse additive turbine statistics before transporting farm summaries.

    Parameters
    ----------
    operation
        Configured farm evaluation.
    parameters
        Operation arguments, including any final turbine reductions.
    plan
        State-reduction operation for each requested field.
    stats
        Mutable statistics mapping; eligible turbine vectors become scalars.
    power_units
        Per-turbine conversion to watts, required for farm yield.

    Returns
    -------
    scalar
        Names already reduced across turbines. Farm-yield conversion is applied
        before summation so mixed turbine power units remain correct.

    Raises
    ------
    ValueError
        If farm yield has no power-unit conversion factors.
    """
    scalar = set()
    choices = parameters.get("turbines_op", {})
    for variable, choice in plan.items():
        target = choices.get(variable)
        if operation in (
            "calc_mean_farm_power",
            "calc_farm_capacity_factor",
            "calc_farm_efficiency",
            "calc_farm_sum",
        ):
            target = "sum"
        elif operation == "calc_farm_mean":
            target = "mean"
        if choice in ("weights", "sum") and target in (
            "sum",
            "weights_sum",
            "mean",
            "weights",
        ):
            value = stats[variable]["value"]
            factor = value.size if target in ("mean", "weights") else 1
            stats[variable] = {"value": np.asarray(np.sum(value) / factor)}
            scalar.add(variable)
    if operation == "calc_farm_yield":
        if power_units is None:
            raise ValueError("Farm yield reduction requires turbine power units")
        for variable in plan:
            stats[variable] = {
                "value": np.asarray(np.sum(stats[variable]["value"] * power_units))
            }
            scalar.add(variable)
    return scalar


def reduce_farm_chunk(
    operation: str,
    parameters: dict[str, Any],
    chunk: dict[str, Any],
    power_units: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return only sufficient statistics and coordinates needed by the result.

    Parameters
    ----------
    operation
        Configured state, turbine or whole-farm evaluation.
    parameters
        Variable selections, reduction choices and duration options.
    chunk
        Farm coordinates and dimensioned fields, read without mutation.
    power_units
        Per-turbine conversion to watts for scalar farm-yield reduction.

    Returns
    -------
    payload
        Reduction plan and statistics without eliminated scientific axes.
        Surviving state/turbine indices, duration endpoints and capacity metadata
        are included only when needed. No scientific arrays are stored on a model.
    """
    plan = reduction_plan(operation, parameters, chunk["data_vars"])
    if FV.WEIGHT in chunk["data_vars"]:
        _validated_field(chunk, FV.WEIGHT)
    turbine_mode = operation in TURBINE_OPERATIONS
    stats = {
        variable: _reduce_field(chunk, variable, choice, turbine_mode)
        for variable, choice in plan.items()
    }
    result: dict[str, Any] = {"plan": plan, "stats": stats}
    if not turbine_mode:
        result["scalar"] = _reduce_scalar_fields(
            operation, parameters, plan, stats, power_units
        )
    if turbine_mode:
        result["states"] = chunk["coords"][FC.STATE]
    else:
        result["turbines"] = (
            chunk["coords"][FC.TURBINE]
            if operation
            in (
                "reduce_states",
                "calc_states_mean",
                "calc_states_sum",
                "calc_states_std",
                "calc_yield",
            )
            else np.empty(0, dtype=int)
        )
    if operation in ("calc_yield", "calc_farm_yield"):
        states = chunk["coords"][FC.STATE]
        result["times"] = states[
            np.unique([0, max(0, len(states) - 2), len(states) - 1])
        ]
    if operation == "calc_farm_capacity_factor" and FV.CAP in chunk["data_vars"]:
        dims, values = chunk["data_vars"][FV.CAP]
        result["capacity"] = np.asarray(
            np.sum(values[0] if FC.STATE in dims else values)
        )
    return result


def merge_farm_chunks(
    operation: str, chunks: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Merge sufficient statistics in engine state-chunk order.

    Parameters
    ----------
    operation
        Evaluation name determining which scientific axis survives.
    chunks
        Nonempty, state-ordered reduced payloads with matching operation plans.

    Returns
    -------
    values
        Variable-to-array mapping. State reductions yield turbine vectors or
        scalars; turbine reductions concatenate their surviving state values.
    index
        Surviving state or turbine coordinates, empty for scalar-only outputs.

    Raises
    ------
    ValueError
        If no farm evaluation chunks were collected.
    """
    if not chunks:
        raise ValueError(f"{operation}: No farm evaluation chunks")
    plan = chunks[0]["plan"]
    turbine_mode = operation in TURBINE_OPERATIONS
    data = {}
    for variable, choice in plan.items():
        if turbine_mode:
            data[variable] = np.concatenate(
                [c["stats"][variable]["value"] for c in chunks]
            )
        else:
            data[variable] = stats_value(
                merge_stats([c["stats"][variable] for c in chunks], choice), choice
            )
    index = (
        np.concatenate([c["states"] for c in chunks])
        if turbine_mode
        else chunks[0]["turbines"]
    )
    return data, index


def reduce_turbine_values(
    values: dict[str, np.ndarray], operations: dict[str, str]
) -> dict[str, Any]:
    """Complete nonadditive turbine reductions after merging state statistics.

    Parameters
    ----------
    values
        Variable-to-turbine-array mapping, read without mutation.
    operations
        Final turbine reduction for each requested variable. ``weights`` divides
        the sum by the turbine count; ``weights_sum`` leaves it unnormalized.

    Returns
    -------
    reduced
        Requested variables mapped to their scalar farm values.

    Raises
    ------
    KeyError
        If a turbine reduction is unsupported.
    """
    result = {}
    for variable, choice in operations.items():
        if choice not in ("weights", "weights_sum", "mean", "sum", "min", "max"):
            raise KeyError(f"Unknown turbine reduction '{choice}' for '{variable}'")
        data = values[variable]
        if choice in ("weights", "weights_sum"):
            result[variable] = np.nansum(data) / (
                len(data) if choice == "weights" else 1
            )
        else:
            result[variable] = stats_value(reduce_array(data, choice), choice)
    return result


def yield_from_mean(
    mean: np.ndarray, times: np.ndarray, units: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    """Convert merged state-weighted power to energy in gigawatt-hours.

    Parameters
    ----------
    mean
        State-weighted turbine power or an already unit-converted farm scalar.
    times
        Ordered state timestamps containing the duration endpoints and final
        step; non-datetime coordinates require an explicit duration.
    units
        Conversion to watts, broadcastable to ``mean``.
    parameters
        Duration options: ``annual``, ``hours`` and optional datetime ``delta_t``.
        Annualization uses 8760 hours; datetime duration includes the final step.

    Returns
    -------
    yield_gwh
        Energy with the same surviving dimensions as ``mean``.

    Raises
    ------
    KeyError
        If explicit hours are supplied for datetime states.
    ValueError
        If non-datetime states have neither explicit hours nor annualization.
    """
    annual = parameters.get("annual", False)
    hours = parameters.get("hours")
    if hours is None and annual:
        duration_hours = 8760.0
    elif np.issubdtype(times.dtype, np.datetime64):
        if hours is not None:
            raise KeyError("Unexpected parameter 'hours' for timeseries data")
        delta = parameters.get("delta_t")
        if delta is None:
            delta = times[-1] - times[-2]
        duration = times[-1] - times[0] + delta
        duration_hours = float(np.int64(duration.astype(np.int64) / 1e9)) / 3600.0
    elif hours is None:
        raise ValueError(
            "Expecting parameter 'hours' for non-timeseries data, or 'annual=True'"
        )
    else:
        duration_hours = float(hours)
    if annual:
        duration_hours = 8760.0
    return mean * duration_hours * units / 1e9
