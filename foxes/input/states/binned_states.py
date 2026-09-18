from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config, get_input_path
from foxes.core import FData, MData, States, TData, WindFarm, Turbine, run_with_engine
from foxes.utils import plot_wind_rose_bars, uv2wd, wd2uv, write_nc

if TYPE_CHECKING:
    from foxes.core import Algorithm, LoadedData, Model


class BinnedStates(States):
    """
    States created by binning another states model at support points.

    The wrapped states are evaluated at the support points during
    initialization. The source states are then reduced into the Cartesian
    product of the configured bins and stored as point-dependent state data.
    During an algorithm calculation, the precomputed bin data is interpolated
    from the support points to the requested target points.

    Source weights must be state-dependent only, as for timeseries input. If
    the source does not provide weights, equal weights are used.
    """

    def __init__(
        self,
        states: States | str | Path | xr.Dataset,
        *,
        bin_vars: Mapping[str, Sequence[float] | int] | None = None,
        support_points: np.ndarray | None = None,
        support_grid: Mapping[str, Sequence[float]] | None = None,
        output_file: str | Path | None = None,
        interpolation: str = "linear",
        fill_value: float | None = np.nan,
        **kwargs: Any,
    ) -> None:
        """
        Initialize binned states.

        Parameters
        ----------
        states
            The source states model to evaluate and bin, or a NetCDF file path
            or Dataset previously written by ``BinnedStates``.
        bin_vars
            Mapping from source variable names to monotonically increasing bin
            edges. For wind speed, an integer specifies the number of equal
            subdivisions between 0 and 30 m/s. For wind direction, an integer
            specifies the number of equal-width bins with the first bin
            centered at 0 degrees. The output state count is the product of
            the number of bins for all variables.
        support_points
            Scattered support coordinates with shape ``(n_points, 3)`` and
            columns ``x``, ``y``, and ``h``. Provide exactly one of
            ``support_points`` and ``support_grid``.
        support_grid
            Regular support-grid axes as a mapping with keys ``x``, ``y``,
            and ``h``. Provide exactly one of ``support_points`` and
            ``support_grid``.
        output_file
            Optional NetCDF file path. When given, the reduced bin data is
            written as soon as it is available, using bin dimensions followed
            by the regular-grid or scattered-point support topology. The file
            can be passed to ``BinnedStates`` as ``states`` in a later run.
        interpolation
            Interpolation method for scattered support points, passed to
            :func:`scipy.interpolate.griddata`.
        fill_value
            Value used for target points outside the support domain.
        kwargs
            Additional arguments for :class:`foxes.core.States`.

        Raises
        ------
        ValueError
            If the bin definitions or support geometry are invalid, or if
            both support representations are provided.
        """
        super().__init__(**kwargs)
        self.states = states if isinstance(states, States) else None
        self.data_source = None if isinstance(states, States) else states
        self.bin_vars: dict[str, np.ndarray] = {}
        if bin_vars is None and self.states is not None:
            raise ValueError(
                "BinnedStates: Require bin_vars when source states are given"
            )
        if bin_vars is not None:
            self._set_bin_vars(bin_vars)
        self.support_points = (
            None
            if support_points is None
            else np.asarray(support_points, dtype=config.dtype_double)
        )
        self.support_grid = (
            None
            if support_grid is None
            else {
                v: np.asarray(values, dtype=config.dtype_double)
                for v, values in support_grid.items()
            }
        )
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.output_file = None if output_file is None else Path(output_file)
        self._bin_shape = tuple(len(edges) - 1 for edges in self.bin_vars.values())
        self._n_bins = int(np.prod(self._bin_shape, dtype=np.int64))
        self._support_key = self.var("support")
        self._grid_axes_key = self.var("grid_axes")
        self._source_state_key = self.var(FC.STATE + "0")
        self._bin_centres_key = self.var("bin_centres")
        self._bin_vars_key = self.var("bin_vars")
        self._cache: dict[str, Any] = {}

        if self.data_source is not None:
            if self.support_points is not None or self.support_grid is not None:
                raise ValueError(
                    "BinnedStates: NetCDF or Dataset input provides its own support topology"
                )
        elif (self.support_points is None) == (self.support_grid is None):
            raise ValueError(
                "BinnedStates: Require exactly one of support_points or support_grid"
            )
        if self.support_points is not None:
            if self.support_points.ndim != 2 or self.support_points.shape[1] != 3:
                raise ValueError(
                    "BinnedStates: support_points must have shape (n_points, 3)"
                )
            if len(self.support_points) == 0:
                raise ValueError("BinnedStates: support_points must not be empty")
        elif self.support_grid is not None:
            assert self.support_grid is not None
            if set(self.support_grid) != {FV.X, FV.Y, FV.H}:
                raise ValueError(
                    f"BinnedStates: support_grid must contain {FV.X}, {FV.Y}, and {FV.H}"
                )
            if any(
                values.ndim != 1 or len(values) == 0
                for values in self.support_grid.values()
            ):
                raise ValueError(
                    "BinnedStates: Grid axes must be non-empty one-dimensional arrays"
                )

    def _set_bin_vars(
        self,
        bin_vars: Mapping[str, Sequence[float] | int],
    ) -> None:
        self.bin_vars = {}
        for var, edges in bin_vars.items():
            if isinstance(edges, (int, np.integer)) and not isinstance(edges, bool):
                if edges < 1:
                    raise ValueError("BinnedStates: Integer bin count must be positive")
                if var == FV.WS:
                    edges = np.linspace(0.0, 30.0, edges + 1, dtype=config.dtype_double)
                elif var == FV.WD:
                    width = 360.0 / edges
                    edges = np.arange(edges + 1, dtype=config.dtype_double) * width
                    edges -= width / 2.0
                else:
                    raise ValueError(
                        f"BinnedStates: Integer bin definitions are only supported for '{FV.WS}' and '{FV.WD}'"
                    )
            self.bin_vars[var] = np.asarray(edges, dtype=config.dtype_double)
        if not self.bin_vars:
            raise ValueError("BinnedStates: Require at least one binned variable")
        for var, edges in self.bin_vars.items():
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
                raise ValueError(f"BinnedStates: Invalid bin edges for '{var}'")
            if np.any(np.diff(edges) <= 0):
                raise ValueError(
                    f"BinnedStates: Bin edges for '{var}' must be increasing"
                )
        self._bin_shape = tuple(len(edges) - 1 for edges in self.bin_vars.values())
        self._n_bins = int(np.prod(self._bin_shape, dtype=np.int64))

    def sub_models(self) -> list[Model]:
        """Return the wrapped source states model."""
        return [] if self.states is None else [self.states]

    def size(self) -> int:
        """Return the number of Cartesian histogram bins."""
        return self._n_bins

    def index(self) -> list[int]:
        """Return positional indices for the histogram bins."""
        return list(range(self._n_bins))

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables represented by the histogram bins.

        Parameters
        ----------
        algo
            The algorithm using this states model.

        Returns
        -------
        list[str]
            The variables configured through ``bin_vars``.
        """
        return list(self.bin_vars)

    def _materialize_support(self) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if self.support_points is not None:
            return self.support_points.copy(), None
        assert self.support_grid is not None
        axes = tuple(self.support_grid[var] for var in (FV.X, FV.Y, FV.H))
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([values.ravel() for values in mesh], axis=-1)
        return points, axes

    def _source_dataset(self, loaded_data: LoadedData) -> xr.Dataset:
        data_vars = {}
        for name, (dims, data) in loaded_data["data_vars"].items():
            if FC.STATE in dims:
                data_vars[name] = (dims, data)
        coords = {
            name: data
            for name, data in loaded_data["coords"].items()
            if isinstance(data, np.ndarray)
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def _read_input_dataset(self) -> xr.Dataset:
        assert self.data_source is not None
        if isinstance(self.data_source, xr.Dataset):
            return self.data_source
        with xr.open_dataset(
            get_input_path(self.data_source), engine=config.nc_engine
        ) as data:
            return data.load()

    @staticmethod
    def _read_input_bin_vars(data: xr.Dataset) -> dict[str, np.ndarray]:
        names = str(data.attrs["foxes_binned_states_bin_vars"]).split(",")
        return {
            name: np.asarray(data.attrs[f"{name}_bounds"], dtype=config.dtype_double)
            for name in names
        }

    @staticmethod
    def _stat_var(var: str, stat: str) -> str:
        return f"{var}_{stat}"

    def _bin_centres(self) -> np.ndarray:
        centres = []
        for var, edges in self.bin_vars.items():
            values = 0.5 * (edges[:-1] + edges[1:])
            if var == FV.WD:
                values = np.mod(values, 360.0)
            centres.append(values)
        mesh = np.meshgrid(*centres, indexing="ij")
        return np.stack([values.ravel() for values in mesh], axis=-1)

    def _reshape_support_data(
        self,
        data: np.ndarray,
        support_shape: tuple[int, ...],
    ) -> np.ndarray:
        return data.reshape(self._bin_shape + support_shape)

    def _create_output_dataset(
        self,
        support: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
    ) -> xr.Dataset:
        bin_coords = {
            var: 0.5 * (edges[:-1] + edges[1:]) for var, edges in self.bin_vars.items()
        }
        if FV.WD in bin_coords:
            bin_coords[FV.WD] = np.mod(bin_coords[FV.WD], 360.0)
        attrs = {
            "foxes_state_class": "BinnedStates",
            "foxes_binned_states_bin_vars": ",".join(self.bin_vars),
        }
        if config.utm_zone_set:
            utm_number, utm_letter = config.utm_zone
            attrs["utm_zone"] = f"{utm_number}{utm_letter}"
        attrs.update({f"{var}_bounds": edges for var, edges in self.bin_vars.items()})

        bin_dims = tuple(self.bin_vars)

        if axes is not None:
            x, y, h = axes
            support_dims: tuple[str, ...] = (FV.X, FV.Y, FV.H)
            support_shape = tuple(len(axis) for axis in axes)
            grid_data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
                self._stat_var(var, stat): (
                    bin_dims + support_dims,
                    self._reshape_support_data(values, support_shape),
                )
                for var, values_by_stat in stats.items()
                for stat, values in values_by_stat.items()
            }
            grid_data_vars[FV.WEIGHT] = (
                bin_dims + support_dims,
                self._reshape_support_data(weights, support_shape),
            )
            return xr.Dataset(
                data_vars=grid_data_vars,
                coords={**bin_coords, FV.X: x, FV.Y: y, FV.H: h},
                attrs=attrs,
            )

        point_data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            self._stat_var(var, stat): (
                bin_dims + (FC.POINT,),
                self._reshape_support_data(values, (support.shape[0],)),
            )
            for var, values_by_stat in stats.items()
            for stat, values in values_by_stat.items()
        }
        point_data_vars[FV.WEIGHT] = (
            bin_dims + (FC.POINT,),
            self._reshape_support_data(weights, (support.shape[0],)),
        )
        return xr.Dataset(
            data_vars=point_data_vars,
            coords={
                **bin_coords,
                FC.POINT: np.arange(support.shape[0], dtype=config.dtype_int),
                FC.XYH: (FC.XYH, np.asarray([FV.X, FV.Y, FV.H])),
                "support": ((FC.POINT, FC.XYH), support),
            },
            attrs=attrs,
        )

    def _load_output_dataset(
        self,
        data: xr.Dataset,
        loaded_data: LoadedData,
    ) -> None:
        bin_dims = tuple(self.bin_vars)
        support_dims: tuple[str, ...]
        if all(var in data.coords for var in (FV.X, FV.Y, FV.H)):
            axes = tuple(np.asarray(data[var].to_numpy()) for var in (FV.X, FV.Y, FV.H))
            support_shape = tuple(len(axis) for axis in axes)
            support_dims = (FV.X, FV.Y, FV.H)
            mesh = np.meshgrid(*axes, indexing="ij")
            support = np.stack([values.ravel() for values in mesh], axis=-1)
        elif "support" in data.coords:
            axes = None
            support = np.asarray(data["support"].to_numpy(), dtype=config.dtype_double)
            support_shape = (support.shape[0],)
            support_dims = (FC.POINT,)
        else:
            raise KeyError("BinnedStates: Missing support coordinates in input data")

        full_dims = bin_dims + support_dims
        loaded_data["coords"][FC.STATE] = np.arange(
            self._n_bins, dtype=config.dtype_int
        )
        loaded_data["coords"][self._bin_vars_key] = np.asarray(list(self.bin_vars))
        loaded_data["data_vars"][self._bin_centres_key] = (
            (FC.STATE, self._bin_vars_key),
            self._bin_centres(),
        )
        for var in self.bin_vars:
            for stat in ["min", "mean", "max"]:
                name = self._stat_var(var, stat)
                if name not in data:
                    raise KeyError(f"BinnedStates: Missing data variable '{name}'")
                values = data[name].transpose(*full_dims).to_numpy()
                loaded_data["data_vars"][self.var(name)] = (
                    (FC.STATE, FC.POINT),
                    values.reshape(self._n_bins, int(np.prod(support_shape))),
                )
        if FV.WEIGHT not in data:
            raise KeyError(f"BinnedStates: Missing data variable '{FV.WEIGHT}'")
        weights = data[FV.WEIGHT].transpose(*full_dims).to_numpy()
        loaded_data["data_vars"][self.var(FV.WEIGHT)] = (
            (FC.STATE, FC.POINT),
            weights.reshape(self._n_bins, int(np.prod(support_shape))),
        )
        loaded_data["extra_data"][self._support_key] = support
        loaded_data["extra_data"][self._grid_axes_key] = axes

    def _write_output_file(
        self,
        support: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
        verbosity: int,
    ) -> None:
        if self.output_file is None:
            return
        write_nc(
            self._create_output_dataset(support, axes, stats, weights),
            self.output_file,
            verbosity=verbosity,
        )

    @staticmethod
    def _circular_bin_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return edges[0] + np.mod(values - edges[0], 360.0)

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Evaluate and reduce the source states during initialization.

        The source is evaluated at all materialized support points through
        the active foxes engine. The resulting arrays are reduced into bins
        and stored in ``loaded_data``. Arrays with the histogram state
        dimension are stored as data variables; support arrays and other
        arrays are stored as extra data.

        Parameters
        ----------
        algo
            The algorithm that owns the active engine and model data.
        loaded_data
            Shared model data populated during initialization.
        force
            Rebuild the source and histogram data when ``True``.
        verbosity
            Initialization verbosity level.

        Raises
        ------
        ValueError
            If source weights are not state-dependent only, or if source
            output data has an unsupported shape.
        """
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        if not force and self._support_key in loaded_data["extra_data"]:
            return

        if self.data_source is not None:
            data = self._read_input_dataset()
            try:
                self._set_bin_vars(self._read_input_bin_vars(data))
                self._load_output_dataset(data, loaded_data)
            finally:
                if data is not self.data_source:
                    data.close()
            return

        support, axes = self._materialize_support()
        assert self.states is not None
        source_states = self.states
        n_states = self.states.size()
        source_data = self._source_dataset(loaded_data)
        from foxes.algorithms import Downwind

        hfarm = WindFarm()
        hfarm.add_turbine(
            Turbine(xy=algo.farm.turbines[0].xy, turbine_models=["null_type"]),
            verbosity=0,
        )
        halgo = Downwind(
            farm=hfarm,
            states=self.states,
            rotor_model="centre",
            partial_wakes="centre",
            wake_models=[],
            verbosity=verbosity - 1,
        )
        halgo.initialize(force=True)

        def _calc_source(halgo: Algorithm = halgo) -> xr.Dataset:
            source_farm_results = halgo.calc_farm()
            source_results = halgo.calc_points(
                source_farm_results,
                support,
                outputs=source_states.output_point_vars(halgo),
            )
            return source_results

        source_results = run_with_engine(_calc_source)
        del halgo, hfarm

        if FV.WEIGHT in source_data:
            weight_dims = source_data[FV.WEIGHT].dims
            if weight_dims != (FC.STATE,):
                raise ValueError(
                    f"BinnedStates: Source weights must be state-dependent only, "
                    f"expecting dimensions {(FC.STATE,)}, got {weight_dims}"
                )
            source_weights = np.asarray(source_data[FV.WEIGHT].to_numpy())
        else:
            source_weights = np.full(
                n_states, 1.0 / n_states, dtype=config.dtype_double
            )
        if source_weights.ndim != 1 or len(source_weights) != n_states:
            raise ValueError(
                "BinnedStates: Source weights must have one value per source state"
            )

        bin_indices = []
        for var, edges in self.bin_vars.items():
            if var not in source_results:
                raise KeyError(
                    f"BinnedStates: Source does not provide binned variable '{var}'"
                )
            values = source_results[var].to_numpy()
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.ndim != 2:
                raise ValueError(
                    f"BinnedStates: Source variable '{var}' must have dimensions (state, point)"
                )
            if var == FV.WD:
                values = self._circular_bin_values(values, edges)
            indices = np.searchsorted(edges, values, side="right") - 1
            bin_indices.append(indices)

        valid = np.ones((n_states, support.shape[0]), dtype=bool)
        flat_bin = np.zeros((n_states, support.shape[0]), dtype=config.dtype_int)
        multiplier = 1
        for indices, nbin in zip(reversed(bin_indices), reversed(self._bin_shape)):
            valid &= (indices >= 0) & (indices < nbin)
            flat_bin += np.where(valid, indices, 0) * multiplier
            multiplier *= nbin

        weights = np.zeros((self._n_bins, support.shape[0]), dtype=config.dtype_double)
        for point_i in range(support.shape[0]):
            mask = valid[:, point_i]
            np.add.at(
                weights[:, point_i],
                flat_bin[mask, point_i],
                source_weights[mask],
            )

        stats: dict[str, dict[str, np.ndarray]] = {}
        for var in self.bin_vars:
            values = source_results[var].to_numpy()
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            mean = np.zeros((self._n_bins, support.shape[0]), dtype=config.dtype_double)
            vmin = np.full(
                (self._n_bins, support.shape[0]), np.inf, dtype=config.dtype_double
            )
            vmax = np.full(
                (self._n_bins, support.shape[0]), -np.inf, dtype=config.dtype_double
            )
            if var == FV.WD:
                wind_vectors = np.zeros(
                    (self._n_bins, support.shape[0], 2), dtype=config.dtype_double
                )
                for point_i in range(support.shape[0]):
                    mask = valid[:, point_i]
                    bins = flat_bin[mask, point_i]
                    weights_i = source_weights[mask]
                    np.add.at(
                        wind_vectors[:, point_i],
                        bins,
                        wd2uv(values[mask, point_i], weights_i),
                    )
                    circular_values = self._circular_bin_values(
                        values[mask, point_i], self.bin_vars[var]
                    )
                    np.minimum.at(vmin[:, point_i], bins, circular_values)
                    np.maximum.at(vmax[:, point_i], bins, circular_values)
                mean[:] = uv2wd(wind_vectors)
                mean[np.isclose(mean, 360.0)] = 0.0
            else:
                for point_i in range(support.shape[0]):
                    mask = valid[:, point_i]
                    bins = flat_bin[mask, point_i]
                    np.add.at(
                        mean[:, point_i],
                        bins,
                        source_weights[mask] * values[mask, point_i],
                    )
                    np.minimum.at(vmin[:, point_i], bins, values[mask, point_i])
                    np.maximum.at(vmax[:, point_i], bins, values[mask, point_i])
            if var != FV.WD:
                np.divide(mean, weights, out=mean, where=weights > 0)
                mean[weights <= 0] = np.nan
            else:
                mean[weights <= 0] = np.nan
            vmin[weights <= 0] = np.nan
            vmax[weights <= 0] = np.nan
            stats[var] = {"min": vmin, "mean": mean, "max": vmax}

        if self._source_state_key not in loaded_data["coords"]:
            source_state = loaded_data["coords"].pop(FC.STATE, None)
            if source_state is not None:
                loaded_data["coords"][self._source_state_key] = source_state
            for name in list(loaded_data["data_vars"]):
                dims, data = loaded_data["data_vars"][name]
                if FC.STATE in dims:
                    loaded_data["data_vars"][name] = (
                        tuple(
                            self._source_state_key if dim == FC.STATE else dim
                            for dim in dims
                        ),
                        data,
                    )

        loaded_data["coords"][FC.STATE] = np.arange(
            self._n_bins, dtype=config.dtype_int
        )
        loaded_data["coords"][self._bin_vars_key] = np.asarray(list(self.bin_vars))
        loaded_data["data_vars"][self._bin_centres_key] = (
            (FC.STATE, self._bin_vars_key),
            self._bin_centres(),
        )
        for var, values_by_stat in stats.items():
            for stat, data in values_by_stat.items():
                loaded_data["data_vars"][self.var(self._stat_var(var, stat))] = (
                    (FC.STATE, FC.POINT),
                    data,
                )
        loaded_data["data_vars"][self.var(FV.WEIGHT)] = (
            (FC.STATE, FC.POINT),
            weights,
        )
        loaded_data["extra_data"][self._support_key] = support
        loaded_data["extra_data"][self._grid_axes_key] = axes
        self._write_output_file(support, axes, stats, weights, verbosity)

    def _get_state_interpolator(
        self,
        var: str,
        support: np.ndarray,
        values: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
    ) -> Any:
        key = f"{var}:{id(values)}"
        if key in self._cache:
            return self._cache[key]
        same_height = np.allclose(support[:, 2], support[0, 2])
        if axes is not None:
            grid_shape = tuple(len(axis) for axis in axes)
            grid_values = values.reshape((values.shape[0],) + grid_shape)
            grid_values = np.moveaxis(grid_values, 0, -1)
            if same_height:
                interpolator = RegularGridInterpolator(
                    axes[:2],
                    grid_values[:, :, 0, :],
                    bounds_error=False,
                    fill_value=self.fill_value,
                )
            else:
                interpolator = RegularGridInterpolator(
                    axes,
                    grid_values,
                    bounds_error=False,
                    fill_value=self.fill_value,
                )
        else:
            coordinates = support[:, :2] if same_height else support
            interp_values = values.T

            def interpolator(points: np.ndarray) -> np.ndarray:
                query = points[:, :2] if same_height else points
                return griddata(
                    coordinates,
                    interp_values,
                    query,
                    method=self.interpolation,
                    fill_value=self.fill_value,
                )

        self._cache[key] = interpolator
        return interpolator

    def _interpolate_states(
        self,
        var: str,
        support: np.ndarray,
        values: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
        points: np.ndarray,
        tdata: TData,
    ) -> np.ndarray:
        interpolator = self._get_state_interpolator(var, support, values, axes)
        same_height = np.allclose(support[:, 2], support[0, 2])
        query = points.reshape(-1, 3)
        if axes is not None and same_height:
            query = query[:, :2]
        interpolated = interpolator(query)
        state_indices = np.repeat(
            np.arange(tdata.n_states, dtype=config.dtype_int),
            tdata.n_targets * tdata.n_tpoints,
        )
        return interpolated[np.arange(interpolated.shape[0]), state_indices].reshape(
            tdata.n_states, tdata.n_targets, tdata.n_tpoints
        )

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, np.ndarray]:
        """
        Interpolate precomputed bins to the current target points.

        Histogram reduction is performed during :meth:`load_data`; this
        method only consumes the histogram-bin chunks supplied through
        ``mdata`` and returns arrays with shape
        ``(n_states, n_targets, n_tpoints)``.

        Parameters
        ----------
        algo
            The algorithm using this states model.
        mdata
            Model data containing the current histogram-bin chunk.
        fdata
            Farm data for the current calculation.
        tdata
            Target coordinates for the current calculation.

        Returns
        -------
        dict[str, numpy.ndarray]
            Interpolated values for the variables configured in ``bin_vars``.
        """
        self.ensure_output_vars(algo, tdata)
        assert tdata.n_states is not None
        assert tdata.n_targets is not None
        assert tdata.n_tpoints is not None
        support = np.asarray(mdata.extra_data[self._support_key])
        axes = mdata.extra_data[self._grid_axes_key]
        points = np.asarray(tdata[FC.TARGETS])
        results = {}
        bin_vars = mdata[self._bin_vars_key].tolist()
        bin_centres = mdata[self._bin_centres_key]
        for var in self.bin_vars:
            if var == FV.WD:
                results[var] = tdata[var]
                results[var][:] = bin_centres[:, bin_vars.index(var), None, None]
            else:
                values = mdata[self.var(self._stat_var(var, "mean"))]
                results[var] = self._interpolate_states(
                    var, support, values, axes, points, tdata
                )
        weights = mdata[self.var(FV.WEIGHT)]
        interpolated_weights = self._interpolate_states(
            FV.WEIGHT, support, weights, axes, points, tdata
        )
        tdata[FV.WEIGHT] = interpolated_weights
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
        return results

    def get_support_wind_rose_data(self, loaded_data: LoadedData) -> xr.Dataset:
        """
        Return wind-rose frequencies at all support points.

        This method only uses data prepared by ``load_data``. It is intended
        to be called while the selected foxes engine is active, before the
        returned data is passed to the figure method outside that context.

        Parameters
        ----------
        loaded_data
            Loaded model data containing the binned support-point weights.

        Returns
        -------
        xarray.Dataset
            Frequencies in percent with dimensions
            ``(point, WD, WS)`` and support-point coordinates. The dataset
            also contains direction and speed bin bounds in its attributes.

        Raises
        ------
        KeyError
            If wind speed or wind direction is not configured as a binned
            variable.
        """
        if FV.WS not in self.bin_vars or FV.WD not in self.bin_vars:
            raise KeyError(
                f"BinnedStates '{self.name}': Wind-rose output requires '{FV.WS}' and '{FV.WD}' bins"
            )

        support = np.asarray(loaded_data["extra_data"][self._support_key])
        weights = loaded_data["data_vars"][self.var(FV.WEIGHT)][1]
        ws_axis = list(self.bin_vars).index(FV.WS)
        wd_axis = list(self.bin_vars).index(FV.WD)
        point_weights = weights.reshape((self._bin_shape) + (support.shape[0],))
        point_weights = np.moveaxis(point_weights, (ws_axis, wd_axis), (0, 1))
        point_weights = np.transpose(point_weights, (2, 1, 0))
        wd_edges = self.bin_vars[FV.WD]
        ws_edges = self.bin_vars[FV.WS]
        return xr.Dataset(
            data_vars={
                "frequency": ((FC.POINT, FV.WD, FV.WS), 100.0 * point_weights),
            },
            coords={
                FC.POINT: np.arange(support.shape[0], dtype=config.dtype_int),
                FC.XYH: (FC.XYH, np.asarray([FV.X, FV.Y, FV.H])),
                FV.WD: 0.5 * (wd_edges[:-1] + wd_edges[1:]),
                FV.WS: 0.5 * (ws_edges[:-1] + ws_edges[1:]),
                "support": ((FC.POINT, FC.XYH), support),
            },
            attrs={
                f"{FV.WD}_bounds": wd_edges,
                f"{FV.WS}_bounds": ws_edges,
            },
        )

    def get_support_wind_roses_figure(
        self,
        data: xr.Dataset,
        *,
        ncols: int = 4,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:
        """
        Create one wind rose per support point on a single canvas.

        Parameters
        ----------
        data
            Dataset returned by :meth:`get_support_wind_rose_data`.
        ncols
            Maximum number of wind-rose axes per row.
        figsize
            Matplotlib figure size. If ``None``, size is derived from the
            number of support points and columns.
        title
            Optional figure-level title.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the support-point wind roses.
        """
        import matplotlib.pyplot as plt

        support = np.asarray(data["support"])
        frequencies = data["frequency"].to_numpy()
        wd_edges = np.asarray(data.attrs[f"{FV.WD}_bounds"])
        n_points = support.shape[0]
        ncols = max(1, min(ncols, n_points))
        nrows = int(np.ceil(n_points / ncols))
        if figsize is None:
            figsize = (4.0 * ncols, 4.0 * nrows)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            subplot_kw={"projection": "polar"},
        )
        for point_i, point in enumerate(support):
            ax = axes.flat[point_i]
            plot_wind_rose_bars(
                ax,
                frequencies[point_i],
                wd_edges,
                cmap="viridis",
            )
            ax.set_title(f"x={point[0]:.0f}, y={point[1]:.0f}, h={point[2]:.0f}")

        for ax in axes.flat[n_points:]:
            ax.set_visible(False)
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    def get_support_wind_rose_fig(
        self,
        data: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        """
        Return a support-point wind-rose figure.

        This compatibility alias delegates to
        :meth:`get_support_wind_roses_figure`.
        """
        return self.get_support_wind_roses_figure(data, **kwargs)

    def write_support_wind_roses(
        self,
        file_name: str,
        data: xr.Dataset,
        *,
        ncols: int = 4,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> None:
        """
        Write a support-point wind-rose canvas to a file.

        Parameters
        ----------
        file_name
            Output image path.
        data
            Dataset returned by :meth:`get_support_wind_rose_data`.
        ncols
            Maximum number of wind-rose axes per row.
        figsize
            Matplotlib figure size. If ``None``, size is derived from the
            number of support points and columns.
        title
            Optional figure-level title.
        """
        fig = self.get_support_wind_roses_figure(
            data, ncols=ncols, figsize=figsize, title=title
        )
        fig.savefig(file_name, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
