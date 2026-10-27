from __future__ import annotations

from collections.abc import Sequence
from glob import glob
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from foxes.algorithms import Downwind
from foxes.config import config, get_input_path, get_output_path
from foxes.core import States, Turbine, WindFarm, run_with_engine
from foxes.input.states.dataset_states import DatasetStates
from foxes.input.states.field_data import FieldData
from foxes.utils import (
    regular_grid_from_points,
    select_grid_axes,
    uv2wd,
    wd2uv,
    write_nc,
)
from foxes.utils.geom2d import AreaGeometry
from foxes.utils.wind_dir import WindDirectionHistogram
import foxes.constants as FC
import foxes.variables as FV


def _axis_values(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and return one spatial coordinate axis."""
    axis = np.asarray(values, dtype=config.dtype_double)
    if axis.ndim != 1 or not len(axis):
        raise ValueError(f"Coordinate '{name}' must be a non-empty 1D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Coordinate '{name}' must contain only finite values")
    if len(axis) > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"Coordinate '{name}' must be strictly increasing")
    return axis


def _prepare_source_data(micro_states: DatasetStates, data: xr.Dataset) -> xr.Dataset:
    """Apply coordinate-relevant source preprocessing."""
    if isinstance(micro_states.sort, bool):
        if micro_states.sort:
            for dim in data.dims:
                if dim in data.coords:
                    data = data.sortby(dim)
    else:
        for dim in micro_states.sort:
            data = data.sortby(dim)
    if micro_states.isel:
        indexers = {
            key: value for key, value in micro_states.isel.items() if key in data.sizes
        }
        data = data.isel(indexers=indexers)
    if micro_states.sel:
        indexers = {
            key: value for key, value in micro_states.sel.items() if key in data.sizes
        }
        data = data.sel(indexers=indexers)
    if micro_states.preprocess_nc is not None:
        data = micro_states.preprocess_nc(data)
    return data


def _source_files(micro_states: DatasetStates) -> list[Path]:
    """Resolve micro-state source files."""
    source = micro_states.data_source
    sources = source if isinstance(source, (list, tuple)) else [source]
    files: list[Path] = []
    for item in sources:
        files.extend(Path(path) for path in glob(str(get_input_path(item))))
    if not files:
        raise FileNotFoundError(f"No micro-state data sources found for '{source}'")
    return sorted(files)


def _extract_source_axes(
    micro_states: FieldData, data: xr.Dataset
) -> dict[str, np.ndarray]:
    """Extract selected Cartesian axes from a field dataset."""
    data = _prepare_source_data(micro_states, data)
    if micro_states.h_coord is None:
        raise ValueError("'micro_states' must define a height coordinate")
    axes = {
        "x": _axis_values("x", data[micro_states.x_coord].to_numpy()),
        "y": _axis_values("y", data[micro_states.y_coord].to_numpy()),
        "height": _axis_values("height", data[micro_states.h_coord].to_numpy()),
    }
    if micro_states.height_bounds is not None:
        lower, upper = micro_states.height_bounds
        heights = axes["height"]
        axes["height"] = heights[(heights >= lower) & (heights <= upper)]
        if not len(axes["height"]):
            raise ValueError(
                f"Micro-state height bounds {micro_states.height_bounds} contain no grid points"
            )
    return axes


def _source_axes(micro_states: FieldData) -> dict[str, np.ndarray]:
    """Read Cartesian axes without loading field variables."""
    source = micro_states.data_source
    if isinstance(source, xr.Dataset):
        return _extract_source_axes(micro_states, source)

    with xr.open_dataset(
        _source_files(micro_states)[0], engine=config.nc_engine
    ) as data:
        return _extract_source_axes(micro_states, data)


def _extract_source_points(micro_states: DatasetStates, data: xr.Dataset) -> np.ndarray:
    """Extract explicit x/y/height support points from a source dataset."""
    data = _prepare_source_data(micro_states, data)
    names = [micro_states.var2ncvar.get(var, var) for var in (FV.X, FV.Y, FV.H)]
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(
            f"Micro states '{micro_states.name}' do not expose x/y/height "
            f"support coordinates; missing {missing}"
        )
    coordinates = [np.asarray(data[name]).reshape(-1) for name in names]
    if len({len(values) for values in coordinates}) != 1:
        raise ValueError(
            f"Micro-state support coordinates have different lengths: "
            f"{dict(zip(names, map(len, coordinates)))}"
        )
    points = np.stack(coordinates, axis=-1).astype(config.dtype_double)
    if not np.all(np.isfinite(points)):
        raise ValueError("Micro-state support points must be finite")
    if micro_states.height_bounds is not None:
        lower, upper = micro_states.height_bounds
        points = points[(points[:, 2] >= lower) & (points[:, 2] <= upper)]
    if not len(points):
        raise ValueError("Micro-state selection contains no support points")
    return points


def _source_points(micro_states: DatasetStates) -> np.ndarray:
    """Read explicit support points without loading field variables."""
    source = micro_states.data_source
    if isinstance(source, xr.Dataset):
        return _extract_source_points(micro_states, source)
    with xr.open_dataset(
        _source_files(micro_states)[0], engine=config.nc_engine
    ) as data:
        return _extract_source_points(micro_states, data)


def _grid_xy(axes: dict[str, np.ndarray]) -> np.ndarray:
    """Create flattened horizontal grid points."""
    x, y = np.meshgrid(axes["x"], axes["y"], indexing="ij")
    return np.stack([x, y], axis=-1).reshape(-1, 2)


def _grid_definition(
    micro_states: DatasetStates,
    boundary: AreaGeometry | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    """Create output axes and return horizontal support for plotting."""
    bounds = (
        None
        if boundary is None
        else (np.asarray(boundary.p_min()), np.asarray(boundary.p_max()))
    )
    if isinstance(micro_states, FieldData):
        source_axes = _source_axes(micro_states)
        selected_axes = select_grid_axes(
            tuple(source_axes.values()), bounds=bounds, padding=1
        )
        axes = dict(zip(source_axes, selected_axes))
        is_cartesian = True
        support_xy = _grid_xy(axes)
    else:
        points = _source_points(micro_states)
        selected_axes, is_cartesian = regular_grid_from_points(
            points, bounds=bounds, padding=1
        )
        axes = dict(zip(("x", "y", "height"), selected_axes))
        support_xy = np.unique(points[:, :2], axis=0)
    if is_cartesian:
        support_xy = _grid_xy(axes)
    return axes, support_xy, is_cartesian


def _grid_axes(
    micro_states: DatasetStates,
    boundary: AreaGeometry | None,
) -> dict[str, np.ndarray]:
    """Create output axes from the micro-state grid and optional boundary."""
    return _grid_definition(micro_states, boundary)[0]


def _write_grid_plot(
    file_path: str | Path,
    axes: dict[str, np.ndarray],
    support_xy: np.ndarray,
    is_cartesian: bool,
    boundary: AreaGeometry | None,
) -> None:
    """Write a proof plot of source support, output grid, and boundary."""
    output_xy = _grid_xy(axes)
    max_points = 250_000

    def _sample(points: np.ndarray) -> np.ndarray:
        step = max(int(np.ceil(len(points) / max_points)), 1)
        return points[::step]

    fig, ax = plt.subplots(figsize=(9, 9))
    if not is_cartesian:
        shown = _sample(support_xy)
        ax.scatter(shown[:, 0], shown[:, 1], s=4, c="0.7", label="Micro-state support")
    shown = _sample(output_xy)
    ax.scatter(shown[:, 0], shown[:, 1], s=2, c="tab:cyan", label="Mean-field grid")
    if boundary is not None:
        boundary.add_to_figure(
            ax,
            show_boundary=True,
            fill_mode=None,
            pars_boundary={"edgecolor": "black", "linewidth": 1.5},
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = get_output_path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _new_algorithm(
    states: States,
    axes: dict[str, np.ndarray],
    verbosity: int,
) -> Downwind:
    """Create a no-wake algorithm for evaluating ambient states."""
    farm = WindFarm()
    endpoints = [(axis[0], axis[-1]) for axis in axes.values()]
    for x, y, height in product(*endpoints):
        farm.add_turbine(
            Turbine(
                xy=[x, y],
                H=height,
                turbine_models=["null_type"],
            ),
            verbosity=0,
        )
    return Downwind(
        farm=farm,
        states=states,
        rotor_model="centre",
        partial_wakes="centre",
        wake_models=[],
        verbosity=verbosity,
    )


def _accumulate(
    results: xr.Dataset,
    variables: list[str],
    point_slice: slice,
    sums: dict[str, np.ndarray],
    weight_sums: dict[str, np.ndarray],
) -> None:
    """Accumulate one state/point result batch."""
    values = {var: results[var] for var in variables if var not in {FV.WS, FV.WD}}
    if FV.WS in variables:
        wind = wd2uv(results[FV.WD].to_numpy(), results[FV.WS].to_numpy())
        values[FV.U] = xr.DataArray(wind[..., 0], dims=results[FV.WS].dims)
        values[FV.V] = xr.DataArray(wind[..., 1], dims=results[FV.WS].dims)

    for var, data in values.items():
        if set(data.dims) != {FC.STATE, FC.POINT}:
            raise ValueError(
                f"Variable '{var}' requires dimensions state and point, got {data.dims}"
            )
        var_weights = results[FV.WEIGHT].broadcast_like(data).transpose(*data.dims)
        weight_data = var_weights.to_numpy()
        if not np.all(np.isfinite(weight_data)) or np.any(weight_data < 0.0):
            raise ValueError("State weights must be finite and non-negative")
        value_data = data.to_numpy()
        valid = np.isfinite(value_data)
        valid_weights = np.where(valid, weight_data, 0.0)
        state_axis = data.get_axis_num(FC.STATE)
        sums[var][point_slice] += np.sum(
            np.where(valid, value_data, 0.0) * valid_weights,
            axis=state_axis,
        )
        weight_sums[var][point_slice] += np.sum(valid_weights, axis=state_axis)


def _main_wd_histogram(
    results: xr.Dataset,
    wd_histo_width: float,
) -> WindDirectionHistogram:
    """Accumulate weighted wind directions from one result batch."""
    directions = results[FV.WD]
    if set(directions.dims) != {FC.STATE, FC.POINT}:
        raise ValueError(
            f"Variable '{FV.WD}' requires dimensions state and point, "
            f"got {directions.dims}"
        )
    weights = results[FV.WEIGHT].broadcast_like(directions).transpose(*directions.dims)
    histogram = WindDirectionHistogram(wd_histo_width)
    histogram.add(
        directions.to_numpy(),
        weights=weights.to_numpy(),
        axis=directions.get_axis_num(FC.STATE),
    )
    return histogram


def _evaluate_mean(
    states: States,
    axes: dict[str, np.ndarray],
    points: np.ndarray,
    variables: list[str],
    states_batch_size: int,
    points_batch_size: int,
    vname_main_wd: str | None,
    wd_histo_width: float,
    verbosity: int,
) -> dict[str, np.ndarray]:
    """Evaluate and reduce states without retaining the state-point cube."""
    mean_vars = [var for var in variables if var not in {FV.WS, FV.WD}]
    if FV.WS in variables:
        mean_vars += [FV.U, FV.V]
    sums = {var: np.zeros(len(points), dtype=config.dtype_double) for var in mean_vars}
    weight_sums = {
        var: np.zeros(len(points), dtype=config.dtype_double) for var in mean_vars
    }
    main_directions = (
        None
        if vname_main_wd is None
        else np.full(len(points), np.nan, dtype=config.dtype_double)
    )
    algo = _new_algorithm(states, axes, verbosity)

    def _calculate() -> None:
        try:
            farm_results = algo.calc_farm(ambient=True, finalize=False)
            n_states = farm_results.sizes[FC.STATE]
            state_index = farm_results.get_index(FC.STATE)
            if not state_index.is_unique:
                raise ValueError(
                    "Mean-field state batching requires unique state labels"
                )
            for point_start in range(0, len(points), points_batch_size):
                point_stop = min(point_start + points_batch_size, len(points))
                point_slice = slice(point_start, point_stop)
                point_histogram: WindDirectionHistogram | None = None
                for state_start in range(0, n_states, states_batch_size):
                    state_stop = min(state_start + states_batch_size, n_states)
                    results = algo.calc_points(
                        farm_results,
                        points[point_slice],
                        outputs=variables + [FV.WEIGHT],
                        ambient=True,
                        ambient_keep=True,
                        finalize=False,
                        states_sel=state_index[state_start:state_stop].tolist(),
                    )
                    missing = [var for var in variables if var not in results]
                    if missing:
                        raise KeyError(
                            f"States do not provide output variables {missing}"
                        )
                    _accumulate(results, variables, point_slice, sums, weight_sums)
                    if main_directions is not None:
                        batch_histogram = _main_wd_histogram(
                            results,
                            wd_histo_width,
                        )
                        if point_histogram is None:
                            point_histogram = batch_histogram
                        else:
                            point_histogram.combine(batch_histogram)
                if main_directions is not None:
                    assert point_histogram is not None
                    main_directions[point_slice] = point_histogram.main_direction()
                if verbosity > 0:
                    print(f"Processed mean-field points: {point_stop}/{len(points)}")
        finally:
            if algo.initialized:
                algo.finalize()

    run_with_engine(_calculate)
    means = {}
    for var in mean_vars:
        if np.any(weight_sums[var] <= 0.0):
            raise ValueError(f"Variable '{var}' has zero total weight")
        means[var] = sums[var] / weight_sums[var]
    if FV.WS in variables:
        uv = np.stack([means.pop(FV.U), means.pop(FV.V)], axis=-1)
        means[FV.WS] = np.linalg.norm(uv, axis=-1)
        means[FV.WD] = uv2wd(uv)
        means[FV.U] = uv[..., 0]
        means[FV.V] = uv[..., 1]
    if main_directions is not None:
        assert vname_main_wd is not None
        means[vname_main_wd] = main_directions
    return means


def create_dataset_mean_from_states(
    states: States,
    micro_states: DatasetStates,
    output_vars: Sequence[str],
    boundary: AreaGeometry | None = None,
    grid_point_plot: str | Path | None = None,
    add_uv: bool = False,
    states_batch_size: int = 100,
    points_batch_size: int = 10000,
    to_file: str | Path | None = None,
    verbosity: int = 1,
    vname_main_wd: str | None = FV.MAIN_WD,
    wd_histo_width: float = 10.0,
) -> xr.Dataset:
    """
    Create a weighted mean field based on micro-state support points.

    Wind speed and direction are averaged as velocity vectors. The optional
    main wind direction is the circular mean of the 50-percent-overlapping
    sector with the greatest total state weight. The returned dataset has no
    state dimension and can be passed directly to
    :class:`foxes.input.states.SingleStateField`. If a boundary is supplied,
    the grid contains its bounding rectangle and one micro-state grid point
    outside each side. Otherwise, the complete micro-state grid is used. For
    non-Cartesian support points, a regular grid is selected within their
    convex hull using the median horizontal nearest-neighbour distance.

    Parameters
    ----------
    states
        The states model to evaluate.
    micro_states
        The dataset states that define the output support points.
    output_vars
        State variables to evaluate and average. Wind speed and direction
        must be requested together.
    boundary
        Optional area geometry that limits the horizontal output grid.
    grid_point_plot
        Optional path for a plot of micro-state support, selected grid points,
        and the boundary.
    add_uv
        Whether to include mean wind-vector components U and V.
    states_batch_size
        Number of states evaluated in each reduction batch.
    points_batch_size
        Number of grid points evaluated in each reduction batch.
    to_file
        Optional NetCDF output path.
    verbosity
        The verbosity level, 0 = silent.
    vname_main_wd
        The variable name for the weighted main wind direction, or ``None``
        to disable its calculation.
    wd_histo_width
        The minimum width of the 50-percent-overlapping main wind direction
        sectors in degrees.

    Returns
    -------
    data
        The weighted mean field with dimensions ``(x, y, height)``.
    """
    axes, support_xy, is_cartesian = _grid_definition(micro_states, boundary)
    variables = list(dict.fromkeys(output_vars))
    if not variables:
        raise ValueError("At least one output variable is required")
    if (FV.WS in variables) != (FV.WD in variables):
        raise ValueError(f"'{FV.WS}' and '{FV.WD}' must be requested together")
    if vname_main_wd is not None:
        if FV.WD not in variables:
            raise ValueError("Main wind direction requires wind speed and direction")
        WindDirectionHistogram(wd_histo_width)
    for name, value in {
        "states_batch_size": states_batch_size,
        "points_batch_size": points_batch_size,
    }.items():
        if value < 1:
            raise ValueError(f"'{name}' must be positive, got {value}")

    mesh = np.meshgrid(*axes.values(), indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, 3)
    if grid_point_plot is not None:
        _write_grid_plot(
            grid_point_plot,
            axes,
            support_xy,
            is_cartesian,
            boundary,
        )
    if verbosity > 0:
        print(
            f"Mean-field grid: {len(axes['x'])} x {len(axes['y'])} x "
            f"{len(axes['height'])}, x={axes['x'][[0, -1]]}, "
            f"y={axes['y'][[0, -1]]}, heights={axes['height']}"
        )
    means = _evaluate_mean(
        states,
        axes,
        points,
        variables,
        states_batch_size,
        points_batch_size,
        vname_main_wd,
        wd_histo_width,
        verbosity,
    )
    if not add_uv:
        means.pop(FV.U, None)
        means.pop(FV.V, None)

    shape = tuple(len(axis) for axis in axes.values())
    dims = tuple(axes)
    data = xr.Dataset(
        coords=axes,
        data_vars={var: (dims, mean.reshape(shape)) for var, mean in means.items()},
    )
    if to_file is not None:
        assert config.nc_engine is not None
        write_nc(data, to_file, nc_engine=config.nc_engine, verbosity=verbosity)
    return data
