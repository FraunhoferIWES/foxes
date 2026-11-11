"""Reduce farm fields with literal state weights and zero-weight NaN validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm


def _get_data(
    results: Dataset, algo: Algorithm | None, output: Output, var: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read a field and identify NaNs allowed by exactly zero state weights.

    Parameters
    ----------
    results
        Read-only farm dataset containing ``var`` and, for NaN fields,
        broadcast-compatible ``FV.WEIGHT``.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output context; not read or mutated.
    var
        Name of the field to validate.

    Returns
    -------
    values
        Field values, potentially sharing input storage; never modified.
    nan_mask
        New boolean array with the field shape. Nonnumeric fields have an
        all-false mask. NaNs require a state axis and exactly zero weights.

    Raises
    ------
    ValueError
        If weights cannot be aligned with the field for NaN validation.
    """
    values = results[var].to_numpy()
    try:
        nan_mask = np.isnan(values)
    except TypeError:
        return values, np.zeros(values.shape, dtype=bool)
    if not np.any(nan_mask):
        return values, nan_mask
    data = results[var]
    assert FC.STATE in data.dims and FV.WEIGHT in results, (
        f"Found {np.sum(nan_mask)} nan values for variable '{var}' of shape {values.shape} without state weights"
    )
    try:
        weights = results[FV.WEIGHT].broadcast_like(data).to_numpy()
    except ValueError as exc:
        raise ValueError(
            f"Cannot align '{FV.WEIGHT}' with variable '{var}' for NaN validation"
        ) from exc
    n_invalid = np.sum(nan_mask & (weights != 0.0))
    assert n_invalid == 0, (
        f"Found {n_invalid} nan values for variable '{var}' of shape {values.shape} with nonzero weights"
    )
    return values, nan_mask


def _validate_nan_results(
    results: Dataset, algo: Algorithm | None, output: Output
) -> None:
    """Validate zero-weight NaN rules for every farm result field.

    Parameters
    ----------
    results
        Read-only farm dataset whose data variables are validated.
    algo
        Algorithm context passed to ``_get_data``.
    output
        Executing output context passed to ``_get_data``.
    """
    for var in results.data_vars:
        _get_data(results, algo, output, var)


def _weinsum(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    rhs: str,
    *vars: str | np.ndarray,
) -> np.ndarray:
    """Contract farm fields with one literal state-weight factor.

    Parameters
    ----------
    results
        Read-only farm dataset with ``FV.WEIGHT`` on ``(FC.STATE,)`` or
        ``(FC.STATE, FC.TURBINE)``.
    algo
        Algorithm context passed to field validation.
    output
        Executing output context passed to field validation.
    rhs
        Output Einstein indices, using ``s`` for states and ``t`` for turbines.
    vars
        Field names or arrays with shape ``(n_states, n_turbines)``. Named
        fields may contain NaNs only at zero weight; array arguments may not.

    Returns
    -------
    reduced
        Newly computed weighted contraction, with axes specified by ``rhs``.
        Allowed named-field NaNs contribute zero; weights are not normalized.

    Raises
    ------
    TypeError
        If an argument is neither a variable name nor a NumPy array.
    ValueError
        If no fields are supplied or weight dimensions are unsupported.
    """
    fields = []
    for var in vars:
        if isinstance(var, str):
            values, nan_mask = _get_data(results, algo, output, var)
            fields.append(np.where(nan_mask, 0.0, values))
        elif isinstance(var, np.ndarray):
            n_nan = np.sum(np.isnan(var))
            assert n_nan == 0, f"Found {n_nan} nan values in an array argument"
            fields.append(var)
        else:
            raise TypeError(f"Expecting variable name as str or array, got {type(var)}")
    if not fields:
        raise ValueError("No data fields supplied for einsum reduction.")
    indices = ["st" for _ in fields]
    weights = results[FV.WEIGHT]
    if weights.dims == (FC.STATE,):
        indices.append("s")
    elif weights.dims == (FC.STATE, FC.TURBINE):
        indices.append("st")
    else:
        raise ValueError(
            f"Expecting '{FV.WEIGHT}' variable with dimensions {(FC.STATE,)} or {(FC.STATE, FC.TURBINE)}, got {weights.dims}"
        )
    fields.append(weights.to_numpy())
    return np.einsum(",".join(indices) + "->" + rhs, *fields)


_NAN_REDUCTIONS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
}


def _reduce_states(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    vars_op: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """Reduce farm states while retaining turbine-coordinate order.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context passed to field validation.
    output
        Executing output context passed to field validation.
    vars_op
        Variable-to-operation mapping: ``"weights"`` uses literal weights;
        ``"mean"``, ``"sum"``, ``"min"``, ``"max"`` and ``"std"`` ignore NaNs;
        ``None`` retains a turbine-only field. When omitted, extensive fields
        are summed, other state-turbine fields are weighted, and turbine-only
        fields are retained.

    Returns
    -------
    reduced
        New DataFrame indexed by ``FC.TURBINE``, with columns in mapping order.
        Allowed zero-weight NaNs are validated before every reduction.

    Raises
    ------
    KeyError
        If a reduction operation is unsupported.
    """
    if vars_op is None:
        vars_op = {
            var: "sum" if var in FV.extensive_state else "weights"
            for var, data in results.data_vars.items()
            if data.dims == (FC.STATE, FC.TURBINE)
        }
        vars_op.update(
            {
                var: None
                for var, data in results.data_vars.items()
                if data.dims == (FC.TURBINE,)
            }
        )
    reduced = {}
    for var, operation in vars_op.items():
        values, _ = _get_data(results, algo, output, var)
        if operation is None:
            reduced[var] = values
        elif operation == "weights":
            reduced[var] = _weinsum(results, algo, output, "t", var)
        elif operation == "std":
            reduced[var] = np.nanstd(values, axis=0)
        elif operation in _NAN_REDUCTIONS:
            reduced[var] = _NAN_REDUCTIONS[operation](values, axis=0)
        else:
            raise KeyError(
                f"Unknown operation '{operation}' for variable '{var}'. Please choose: weights, mean, sum, min, max"
            )
    frame = pd.DataFrame(index=results[FC.TURBINE].values, data=reduced)
    frame.index.name = FC.TURBINE
    return frame


def _reduce_turbines(
    results: Dataset, algo: Algorithm | None, output: Output, vars_op: dict[str, str]
) -> pd.DataFrame:
    """Reduce turbine fields while retaining state-coordinate order.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context passed to field validation.
    output
        Executing output context passed to field validation.
    vars_op
        Variable-to-operation mapping. ``"weights_sum"`` sums weighted turbine
        values; ``"weights"`` also divides by turbine count, not weight sum.
        ``"mean"``, ``"sum"``, ``"min"`` and ``"max"`` ignore allowed NaNs.

    Returns
    -------
    reduced
        New DataFrame indexed by ``FC.STATE`` with columns in mapping order.

    Raises
    ------
    KeyError
        If a reduction operation is unsupported.
    """
    reduced = {}
    for var, operation in vars_op.items():
        values, _ = _get_data(results, algo, output, var)
        if operation in ("weights", "weights_sum"):
            reduced[var] = _weinsum(results, algo, output, "s", var)
            if operation == "weights":
                reduced[var] = reduced[var] / values.shape[1]
        elif operation in _NAN_REDUCTIONS:
            reduced[var] = _NAN_REDUCTIONS[operation](values, axis=1)
        else:
            raise KeyError(
                f"Unknown operation '{operation}' for variable '{var}'. Please choose: weights, weights_sum, mean, sum, min, max"
            )
    frame = pd.DataFrame(index=results.coords[FC.STATE].to_numpy(), data=reduced)
    frame.index.name = FC.STATE
    return frame


def _reduce_all(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    states_op: dict[str, str | None],
    turbines_op: dict[str, str],
) -> dict:
    """Reduce states and then turbines without normalizing state weights.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context passed to field validation.
    output
        Executing output context passed to field validation.
    states_op
        Variable-to-state-operation mapping accepted by ``_reduce_states``.
    turbines_op
        Variable-to-turbine-operation mapping. ``"weights"`` divides by turbine
        count; ``"weights_sum"`` does not. Other choices are ``"mean"``,
        ``"sum"``, ``"min"`` and ``"max"``.

    Returns
    -------
    reduced
        New variable-to-scalar mapping in ``turbines_op`` order.

    Raises
    ------
    KeyError
        If an operation is unsupported or a turbine field was not state-reduced.
    """
    state_data = _reduce_states(results, algo, output, states_op)
    reduced = {}
    for var, operation in turbines_op.items():
        values = state_data[var].to_numpy()
        if operation in {"weights", "weights_sum"}:
            reduced[var] = (
                _weinsum(results, algo, output, "", var)
                if states_op[var] == "weights"
                else np.nansum(values)
            )
            if operation == "weights":
                reduced[var] /= len(values)
        elif operation in _NAN_REDUCTIONS:
            reduced[var] = _NAN_REDUCTIONS[operation](values)
        else:
            raise KeyError(
                f"Unknown operation '{operation}' for variable '{var}'. Please choose: weights, weights_sum, mean, sum, min, max"
            )
    return reduced


def _calc_states_mean(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    vars: str | list[str],
    use_weights: bool = True,
) -> pd.DataFrame:
    """Calculate turbine means over states using the requested weighting.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Variable name or ordered list of names.
    use_weights
        Use literal, unnormalized state weights rather than an arithmetic mean.

    Returns
    -------
    means
        New DataFrame indexed by ``FC.TURBINE``. Zero-weight NaNs are excluded.
    """
    operation = "weights" if use_weights else "mean"
    variables = [vars] if isinstance(vars, str) else vars
    return _reduce_states(results, algo, output, {var: operation for var in variables})


def _calc_states_sum(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> pd.DataFrame:
    """Sum states separately for each turbine without applying weights.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    sums
        New DataFrame indexed by ``FC.TURBINE``; allowed NaNs are ignored.
    """
    return _reduce_states(results, algo, output, {var: "sum" for var in vars})


def _calc_states_std(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> pd.DataFrame:
    """Calculate population standard deviations over states for each turbine.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    deviations
        New DataFrame indexed by ``FC.TURBINE``. Allowed NaNs are ignored;
        state weights are not used in the standard deviation.
    """
    return _reduce_states(results, algo, output, {var: "std" for var in vars})


def _calc_turbine_mean(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> pd.DataFrame:
    """Calculate arithmetic turbine means separately for each state.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    means
        New DataFrame indexed by ``FC.STATE``; allowed NaNs are ignored.
    """
    return _reduce_turbines(results, algo, output, {var: "mean" for var in vars})


def _calc_turbine_sum(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> pd.DataFrame:
    """Sum turbine values separately for each state without applying weights.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    sums
        New DataFrame indexed by ``FC.STATE``; allowed NaNs are ignored.
    """
    return _reduce_turbines(results, algo, output, {var: "sum" for var in vars})


def _calc_farm_mean(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> dict[str, np.ndarray | float]:
    """Calculate literal-state-weighted means across all turbines.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    means
        New variable-to-scalar mapping; divide weighted totals by turbine count.
    """
    return _reduce_all(
        results,
        algo,
        output,
        states_op={var: "weights" for var in vars},
        turbines_op={var: "weights" for var in vars},
    )


def _calc_farm_sum(
    results: Dataset, algo: Algorithm | None, output: Output, vars: list[str]
) -> dict[str, np.ndarray | float]:
    """Sum state and turbine values without applying weights.

    Parameters
    ----------
    results
        Read-only farm dataset with ``(FC.STATE, FC.TURBINE)`` fields.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    vars
        Ordered variable names.

    Returns
    -------
    sums
        New variable-to-scalar mapping; allowed zero-weight NaNs are ignored.
    """
    return _reduce_all(
        results,
        algo,
        output,
        states_op={var: "sum" for var in vars},
        turbines_op={var: "sum" for var in vars},
    )


def _calc_mean_farm_power(
    results: Dataset, algo: Algorithm | None, output: Output, ambient: bool = False
) -> float:
    """Calculate the literal-state-weighted total farm power.

    Parameters
    ----------
    results
        Read-only farm results with power on ``(FC.STATE, FC.TURBINE)``.
    algo
        Algorithm context for validation.
    output
        Executing output context for validation.
    ambient
        Use ``FV.AMB_P`` instead of ``FV.P``.

    Returns
    -------
    mean_power
        Weighted total power in the source field's units; state weights are
        not normalized and turbine values are summed.
    """
    variable = FV.AMB_P if ambient else FV.P
    return _reduce_all(
        results,
        algo,
        output,
        states_op={variable: "weights"},
        turbines_op={variable: "sum"},
    )[variable]
