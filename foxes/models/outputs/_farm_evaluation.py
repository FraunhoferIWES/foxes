"""Calculate farm performance metrics and expose the shared reduction functions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.core import Output
from foxes.utils import write_nc

from ._farm_statistics import (
    _calc_farm_mean as _calc_farm_mean,
    _calc_farm_sum as _calc_farm_sum,
    _calc_mean_farm_power as _calc_mean_farm_power,
    _calc_states_mean as _calc_states_mean,
    _calc_states_std as _calc_states_std,
    _calc_states_sum as _calc_states_sum,
    _calc_turbine_mean as _calc_turbine_mean,
    _calc_turbine_sum as _calc_turbine_sum,
    _get_data as _get_data,
    _reduce_all as _reduce_all,
    _reduce_states as _reduce_states,
    _reduce_turbines as _reduce_turbines,
    _validate_nan_results as _validate_nan_results,
    _weinsum as _weinsum,
)

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm


def _get_power_units(
    results: Dataset | None, algo: Algorithm | None, output: Output
) -> np.ndarray:
    """Resolve each turbine type's power conversion to watts.

    Parameters
    ----------
    results
        Farm result context; not read or mutated.
    algo
        Initialized algorithm supplying controller turbine types.
    output
        Executing output context; not read or mutated.

    Returns
    -------
    power_units
        New watt conversion factors in turbine order, shape ``(n_turbines,)``.

    Raises
    ------
    KeyError
        If no algorithm is supplied or a turbine power unit is unknown.
    """
    if algo is None:
        raise KeyError("Algorithm object is required for getting power units")
    turbine_types = algo.farm_controller.turbine_types
    assert turbine_types is not None
    return np.array(
        [FC.P_UNITS[turbine.P_unit] for turbine in turbine_types],
        dtype=config.dtype_double,
    )


def _duration_hours(
    results: Dataset, annual: bool, hours: int | None, delta_t: np.timedelta64 | None
) -> float:
    """Resolve the integration duration using the existing time-series convention.

    Parameters
    ----------
    results
        Read-only farm results with an ``FC.STATE`` coordinate.
    annual
        Use 8760 hours when ``hours`` is not explicitly supplied.
    hours
        Explicit duration for non-time-series states.
    delta_t
        Final time step; ``None`` uses the final two datetime state labels.

    Returns
    -------
    duration
        Duration in hours, including the final time step for datetime states.

    Raises
    ------
    KeyError
        If explicit hours are supplied with datetime states.
    ValueError
        If non-datetime states have neither explicit hours nor annual duration.
    """
    if hours is None and annual:
        return 8760.0
    if np.issubdtype(results[FC.STATE].dtype, np.datetime64):
        if hours is not None:
            raise KeyError("Unexpected parameter 'hours' for timeseries data")
        times = results[FC.STATE].to_numpy()
        if delta_t is None:
            delta_t = times[-1] - times[-2]
        duration = times[-1] - times[0] + delta_t
        seconds = np.int64(duration.astype(np.int64) / 1000000000.0)
        return float(seconds) / 3600.0
    if hours is None:
        raise ValueError(
            "Expecting parameter 'hours' for non-timeseries data, or 'annual=True'"
        )
    return float(hours)


def _calc_yield(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    annual: bool = False,
    ambient: bool = False,
    hours: int | None = None,
    delta_t: np.timedelta64 | None = None,
    P_unit_W: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate turbine energy yield from literal-state-weighted power.

    Parameters
    ----------
    results
        Read-only farm power fields on ``(FC.STATE, FC.TURBINE)``.
    algo
        Initialized algorithm supplying power units, exclusive with
        ``P_unit_W``.
    output
        Executing output context passed to reductions.
    annual
        Report annualized yield using 8760 hours per year.
    ambient
        Use ``FV.AMB_P`` and return ``FV.AMB_YLD`` rather than ``FV.YLD``.
    hours
        Duration for non-datetime states. Required unless ``annual`` is enabled.
    delta_t
        Final time step for datetime states; ``None`` uses the final interval.
    P_unit_W
        Watt conversion factors broadcast against the turbine-indexed power
        DataFrame. Supply only when ``algo`` is ``None``.

    Returns
    -------
    turbine_yield
        New DataFrame indexed by ``FC.TURBINE`` with one yield column in GWh,
        or GWh/a for annualized results. State weights are not normalized.

    Raises
    ------
    KeyError
        If both or neither power-unit sources are supplied, or datetime states
        are combined with explicit hours.
    ValueError
        If non-datetime states lack a duration and annualization is disabled.
    """
    var_in, var_out = (FV.AMB_P, FV.AMB_YLD) if ambient else (FV.P, FV.YLD)
    if algo is not None and P_unit_W is None:
        P_unit_W = _get_power_units(results, algo, output)[:, None]
    elif not (algo is None and P_unit_W is not None):
        raise KeyError("Expecting either algorithm or 'P_unit_W'")
    duration = _duration_hours(results, annual, hours, delta_t)
    turbine_yield = (
        _calc_states_mean(results, algo, output, var_in)
        * duration
        * P_unit_W
        / 1000000000.0
    )
    if duration != 8760 and annual:
        turbine_yield *= 8760 / duration
    turbine_yield.rename(columns={var_in: var_out}, inplace=True)
    return turbine_yield


def _get_capacity(
    results: Dataset, algo: Algorithm | None, output: Output
) -> np.ndarray:
    """Read nominal turbine powers from the algorithm's farm.

    Parameters
    ----------
    results
        Farm result context; not read or mutated.
    algo
        Initialized algorithm required to resolve nominal powers.
    output
        Executing output context; not read or mutated.

    Returns
    -------
    capacity
        Nominal powers in turbine order, shape ``(n_turbines,)``, using each
        turbine type's power units.
    """
    assert algo is not None, (
        "Algorithm object is required for adding capacity to farm results"
    )
    return algo.farm.get_capacity_array(algo)


def _add_capacity(
    results: Dataset, algo: Algorithm | None, output: Output, verbosity: int = 1
) -> Dataset:
    """Add nominal turbine powers to the supplied farm results in place.

    Parameters
    ----------
    results
        Dataset receiving ``FV.CAP`` on ``(FC.TURBINE,)``.
    algo
        Initialized algorithm supplying nominal powers.
    output
        Executing output context passed to capacity lookup.
    verbosity
        Print an addition message above zero.

    Returns
    -------
    updated
        The supplied dataset, with ``FV.CAP`` added or replaced.
    """
    results[FV.CAP] = ((FC.TURBINE,), _get_capacity(results, algo, output))
    if verbosity > 0:
        print("Capacity added to farm results")
    return results


def _add_yield(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    annual: bool = True,
    ambient: bool = False,
    verbosity: int = 1,
    **kwargs: Any,
) -> Dataset:
    """Add turbine energy yield to the supplied farm results in place.

    Parameters
    ----------
    results
        Dataset receiving ``FV.YLD`` or ``FV.AMB_YLD`` on ``(FC.TURBINE,)``.
    algo
        Algorithm supplying power units unless explicit units are forwarded.
    output
        Executing output context passed to yield calculation.
    annual
        Annualize yield to GWh/a rather than GWh.
    ambient
        Calculate yield from ``FV.AMB_P`` instead of ``FV.P``.
    verbosity
        Print an addition message above zero.
    kwargs
        Duration and power-unit options forwarded to ``_calc_yield``.

    Returns
    -------
    updated
        The supplied dataset, with the turbine yield field added or replaced.

    Raises
    ------
    KeyError
        If power-unit sources conflict or datetime states receive explicit hours.
    ValueError
        If non-datetime states lack a duration and annualization is disabled.
    """
    turbine_yield = _calc_yield(
        results, algo, output, annual=annual, ambient=ambient, **kwargs
    )
    assert len(turbine_yield.columns) == 1, "Expecting single column in yield dataframe"
    variable = turbine_yield.columns[0]
    results[variable] = ((FC.TURBINE,), turbine_yield[variable].to_numpy())
    if verbosity > 0:
        print(f"{'Ambient yield' if ambient else 'Yield'} added to results")
    return results


def _add_capacity_factor(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    capacity: np.ndarray | None = None,
    ambient: bool = False,
    verbosity: int = 1,
) -> Dataset:
    """Add power divided by nominal turbine capacity to farm results in place.

    Parameters
    ----------
    results
        Dataset receiving ``FV.CAPF`` or ``FV.AMB_CAPF`` on the power dimensions.
        Existing ``FV.CAP`` is used when present.
    algo
        Algorithm supplying capacity when ``FV.CAP`` is absent.
    output
        Executing output context passed to capacity lookup.
    capacity
        Accepted argument; capacity is resolved from ``results`` or ``algo``.
    ambient
        Use ambient power rather than waked power.
    verbosity
        Print an addition message above zero.

    Returns
    -------
    updated
        The supplied dataset, with dimensionless capacity factors added or
        replaced. Division is not clipped and zero capacities are not masked.
    """
    var_in, var_out = (FV.AMB_P, FV.AMB_CAPF) if ambient else (FV.P, FV.CAPF)
    cap = (
        results[FV.CAP].to_numpy()
        if FV.CAP in results
        else _get_capacity(results, algo, output)
    )
    results[var_out] = results[var_in] / cap[None, :]
    if verbosity > 0:
        print(
            f"{'Ambient capacity' if ambient else 'Capacity'} factor added to farm results"
        )
    return results


def _calc_farm_capacity(
    results: Dataset, algo: Algorithm | None, output: Output
) -> float:
    """Sum nominal turbine powers over the farm.

    Parameters
    ----------
    results
        Farm result context passed to capacity lookup.
    algo
        Initialized algorithm supplying nominal powers.
    output
        Executing output context passed to capacity lookup.

    Returns
    -------
    capacity
        Total nominal power in the turbine types' power units.
    """
    return np.sum(_get_capacity(results, algo, output))


def _calc_farm_capacity_factor(
    results: Dataset, algo: Algorithm | None, output: Output, ambient: bool = False
) -> float:
    """Divide literal-state-weighted total power by positive farm capacity.

    Parameters
    ----------
    results
        Read-only farm dataset. Existing ``FV.CAP`` is used at the first state
        label when it includes a state axis.
    algo
        Algorithm supplying nominal capacity when ``FV.CAP`` is absent.
    output
        Executing output context passed to reductions.
    ambient
        Use ``FV.AMB_P`` rather than ``FV.P``.

    Returns
    -------
    capacity_factor
        Dimensionless weighted total power divided by total capacity.

    Raises
    ------
    ValueError
        If total farm capacity is nonpositive.
    """
    if FV.CAP in results:
        capacity_data = results[FV.CAP]
        if FC.STATE in capacity_data.dims:
            label = capacity_data[FC.STATE].to_numpy()[0]
            capacity_data = capacity_data.sel({FC.STATE: label})
        capacity = np.sum(capacity_data.to_numpy())
    else:
        capacity = _calc_farm_capacity(results, algo, output)
    if capacity <= 0.0:
        raise ValueError("Farm capacity must be positive")
    return _calc_mean_farm_power(results, algo, output, ambient=ambient) / capacity


def _calc_farm_yield(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    turbine_yield: pd.DataFrame | None = None,
    power_uncert: float | None = None,
    ambient: bool = False,
    **kwargs: Any,
) -> float | tuple[float, float, float]:
    """Sum turbine yields and optionally estimate farm P75 and P90 yields.

    Parameters
    ----------
    results
        Read-only farm results used when turbine yields are not supplied.
    algo
        Algorithm supplying power units for yield calculation.
    output
        Executing output context passed to yield calculation.
    turbine_yield
        Turbine-indexed DataFrame containing ``FV.YLD`` or ``FV.AMB_YLD``;
        ``None`` calculates annual yield first.
    power_uncert
        Relative power uncertainty; ``None`` omits probability estimates.
    ambient
        Sum ambient yield rather than waked yield.
    kwargs
        Options for ``_calc_yield``, used only when ``turbine_yield`` is absent.

    Returns
    -------
    yield_estimate
        Total yield, or ``(total, P75, P90)`` when uncertainty is supplied,
        in the supplied turbine yield units. Probability estimates multiply
        total yield by ``1 - 0.675 * power_uncert`` and
        ``1 - 1.282 * power_uncert``, respectively.

    Raises
    ------
    KeyError
        If the selected yield column is absent, or calculated yields have
        conflicting power-unit sources or datetime duration options.
    ValueError
        If calculated yields require a duration that was not supplied.
    """
    if turbine_yield is None:
        options: dict[str, Any] = dict(annual=True)
        options.update(kwargs)
        turbine_yield = _calc_yield(results, algo, output, ambient=ambient, **options)
    farm_yield = turbine_yield.sum()
    variable = FV.AMB_YLD if ambient else FV.YLD
    if power_uncert is not None:
        p75 = farm_yield * (1.0 - 0.675 * power_uncert)
        p90 = farm_yield * (1.0 - 1.282 * power_uncert)
        return farm_yield[variable], p75[variable], p90[variable]
    return farm_yield[variable]


def _add_efficiency(
    results: Dataset, algo: Algorithm | None, output: Output, verbosity: int = 1
) -> Dataset:
    """Add clipped turbine power efficiencies to farm results in place.

    Parameters
    ----------
    results
        Dataset containing ``FV.P`` and ``FV.AMB_P`` on matching dimensions.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output context; not read or mutated.
    verbosity
        Print an addition message above zero.

    Returns
    -------
    updated
        The supplied dataset with ``FV.EFF`` added or replaced on ambient-power
        dimensions. The denominator is floored at ``1e-12``, the ratio capped
        at one, and efficiency set to zero where power is below ``1e-10``.
    """
    power = results[FV.P].to_numpy()
    ambient_power = np.maximum(results[FV.AMB_P].to_numpy(), 1e-12)
    efficiency = np.minimum(power / ambient_power, 1)
    efficiency[power < 1e-10] = 0
    results[FV.EFF] = (results[FV.AMB_P].dims, efficiency)
    if verbosity > 0:
        print("Efficiency added to farm results")
    return results


def _calc_farm_efficiency(
    results: Dataset, algo: Algorithm | None, output: Output
) -> float:
    """Divide weighted farm power by weighted ambient farm power.

    Parameters
    ----------
    results
        Read-only farm dataset with ``FV.P`` and ``FV.AMB_P``.
    algo
        Algorithm context passed to reductions.
    output
        Executing output context passed to reductions.

    Returns
    -------
    efficiency
        Dimensionless ratio capped at one, with ambient power floored at
        ``1e-14``. State weights remain literal and unnormalized.
    """
    power = _calc_mean_farm_power(results, algo, output)
    ambient_power = np.maximum(
        _calc_mean_farm_power(results, algo, output, ambient=True), 1e-14
    )
    return np.minimum(power / ambient_power, 1)


def _gen_stdata(
    results: Dataset,
    algo: Algorithm | None,
    output: Output,
    turbines: list[int],
    variable: str,
    fig: Figure | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    legloc: str = "lower right",
    animated: bool = True,
    ret_im: bool = True,
) -> Iterator[Figure | tuple[Figure, list[Line2D]]]:
    """Generate progressively revealed turbine histories for animation.

    Parameters
    ----------
    results
        Read-only farm results with ``variable`` on ``(FC.STATE, FC.TURBINE)``.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output context; not read or mutated.
    turbines
        Positional turbine indices in plotting order.
    variable
        Field to plot against positional state counts.
    fig
        Figure receiving a new subplot when ``ax`` is absent.
    ax
        Caller-owned axes receiving lines without clearing existing artists.
    figsize
        Figure size in inches when creating a figure.
    legloc
        Matplotlib legend location.
    animated
        Mark created lines for animation.
    ret_im
        Include each frame's line artists with its figure.

    Yields
    ------
    frame
        Figure, or ``(figure, lines)`` when ``ret_im`` is enabled, for each
        state. Frame ``si`` shows ``data[:si]``; the first history is empty.
        Figures are not saved, shown or closed.
    """
    figure = plt.figure(figsize=figsize) if fig is None else fig
    axes = figure.add_subplot(111) if ax is None else ax
    axes.set_xlabel("State")
    axes.set_ylabel(variable)
    colors = cycler(color="bgrcmyk")
    values = results[variable].to_numpy()
    labeled = set()
    for si in range(len(values)):
        artists = []
        axes.set_prop_cycle(colors)
        for ti in turbines:
            label = None if ti in labeled else f"Turbine {ti}"
            artists += axes.plot(
                range(si), values[:si, ti], label=label, animated=animated
            )
            labeled.add(ti)
        axes.legend(loc=legloc)
        yield (figure, artists) if ret_im else figure


def _write_nc(
    results: Dataset, algo: Algorithm | None, output: Output, fname: str, **kwargs: Any
) -> Dataset:
    """Write evaluated farm results to a resolved NetCDF path.

    Parameters
    ----------
    results
        Evaluated dataset to write; not modified.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output resolving ``fname`` through ``get_fpath``.
    fname
        Output file name, relative to the output directory unless absolute.
    kwargs
        Options forwarded to ``foxes.utils.write_nc``.

    Returns
    -------
    written
        The supplied dataset, after the write succeeds.
    """
    write_nc(results, output.get_fpath(fname), **kwargs)
    return results
