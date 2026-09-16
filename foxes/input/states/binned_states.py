from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.core import FData, MData, States, TData, launch_parallel_calc
from foxes.utils import plot_wind_rose_bars

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
        states: States,
        *,
        bin_vars: Mapping[str, Sequence[float]],
        support_points: np.ndarray | None = None,
        support_grid: Mapping[str, Sequence[float]] | None = None,
        interpolation: str = "linear",
        fill_value: float | None = np.nan,
        **kwargs: Any,
    ) -> None:
        """
        Initialize binned states.

        Parameters
        ----------
        states
            The source states model to evaluate and bin.
        bin_vars
            Mapping from source variable names to monotonically increasing bin
            edges. The output state count is the product of the number of bins
            for all variables.
        support_points
            Scattered support coordinates with shape ``(n_points, 3)`` and
            columns ``x``, ``y``, and ``h``. Provide exactly one of
            ``support_points`` and ``support_grid``.
        support_grid
            Regular support-grid axes as a mapping with keys ``x``, ``y``,
            and ``h``. Provide exactly one of ``support_points`` and
            ``support_grid``.
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
        self.states = states
        self.bin_vars = {
            v: np.asarray(edges, dtype=config.dtype_double)
            for v, edges in bin_vars.items()
        }
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
        self._bin_shape = tuple(len(edges) - 1 for edges in self.bin_vars.values())
        self._n_bins = int(np.prod(self._bin_shape, dtype=np.int64))
        self._support_key = self.var("support")
        self._grid_axes_key = self.var("grid_axes")
        self._source_state_key = self.var(FC.STATE + "0")
        self._cache: dict[str, Any] = {}

        if not self.bin_vars:
            raise ValueError("BinnedStates: Require at least one binned variable")
        for var, edges in self.bin_vars.items():
            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
                raise ValueError(f"BinnedStates: Invalid bin edges for '{var}'")
            if np.any(np.diff(edges) <= 0):
                raise ValueError(f"BinnedStates: Bin edges for '{var}' must be increasing")
        if (self.support_points is None) == (self.support_grid is None):
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
        else:
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

    def sub_models(self) -> list[Model]:
        """Return the wrapped source states model."""
        return [self.states]

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

        support, axes = self._materialize_support()
        n_states = self.states.size()
        point_data = algo.new_point_data(support, n_states=n_states)
        source_data = self._source_dataset(loaded_data)
        source_vars = self.states.output_point_vars(algo)
        previous_n_states = getattr(algo, "n_states", None)
        if previous_n_states is None:
            algo.n_states = n_states
        try:
            source_results = launch_parallel_calc(
                algo,
                model=self.states,
                model_data=source_data,
                point_data=point_data,
                out_vars=source_vars,
            )
        finally:
            algo.n_states = previous_n_states

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

        weights = np.zeros(
            (self._n_bins, support.shape[0]), dtype=config.dtype_double
        )
        for point_i in range(support.shape[0]):
            mask = valid[:, point_i]
            np.add.at(
                weights[:, point_i],
                flat_bin[mask, point_i],
                source_weights[mask],
            )

        reduced: dict[str, np.ndarray] = {}
        for var in self.bin_vars:
            values = source_results[var].to_numpy()
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            out = np.zeros(
                (self._n_bins, support.shape[0]), dtype=config.dtype_double
            )
            if var == FV.WD:
                sine = np.zeros_like(out)
                cosine = np.zeros_like(out)
                for point_i in range(support.shape[0]):
                    mask = valid[:, point_i]
                    angles = np.deg2rad(values[mask, point_i])
                    bins = flat_bin[mask, point_i]
                    weights_i = source_weights[mask]
                    np.add.at(sine[:, point_i], bins, weights_i * np.sin(angles))
                    np.add.at(cosine[:, point_i], bins, weights_i * np.cos(angles))
                out[:] = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
                out[np.isclose(out, 360.0)] = 0.0
            else:
                for point_i in range(support.shape[0]):
                    mask = valid[:, point_i]
                    np.add.at(
                        out[:, point_i],
                        flat_bin[mask, point_i],
                        source_weights[mask] * values[mask, point_i],
                    )
            if var != FV.WD:
                np.divide(out, weights, out=out, where=weights > 0)
                out[weights <= 0] = np.nan
            else:
                out[weights <= 0] = np.nan
            reduced[var] = out

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
        for var, data in reduced.items():
            loaded_data["data_vars"][self.var(var)] = ((FC.STATE, FC.POINT), data)
        loaded_data["data_vars"][self.var(FV.WEIGHT)] = (
            (FC.STATE, FC.POINT),
            weights,
        )
        loaded_data["extra_data"][self._support_key] = support
        loaded_data["extra_data"][self._grid_axes_key] = axes

    def _get_interpolator(
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
            if same_height:
                grid_values = values.reshape(grid_shape)
                interpolator = RegularGridInterpolator(
                    axes[:2],
                    grid_values[:, :, 0],
                    bounds_error=False,
                    fill_value=self.fill_value,
                )
            else:
                grid_values = values.reshape(grid_shape)
                interpolator = RegularGridInterpolator(
                    axes,
                    grid_values,
                    bounds_error=False,
                    fill_value=self.fill_value,
                )
        else:
            coordinates = support[:, :2] if same_height else support

            def interpolator(points: np.ndarray) -> np.ndarray:
                query = points[:, :2] if same_height else points
                return griddata(
                    coordinates,
                    values,
                    query,
                    method=self.interpolation,
                    fill_value=self.fill_value,
                )

        self._cache[key] = interpolator
        return interpolator

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
        same_height = np.allclose(support[:, 2], support[0, 2])
        results = {}
        for var in self.bin_vars:
            values = mdata[self.var(var)]
            interpolated = np.empty(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                dtype=config.dtype_double,
            )
            for state_i in range(tdata.n_states):
                interpolator = self._get_interpolator(
                    f"{var}:{state_i}",
                    support,
                    values[state_i],
                    axes,
                )
                query = points[state_i].reshape(-1, 3)
                if axes is not None and same_height:
                    query = query[:, :2]
                interpolated[state_i] = interpolator(query).reshape(
                    tdata.n_targets, tdata.n_tpoints
                )
            results[var] = interpolated
        weights = mdata[self.var(FV.WEIGHT)]
        interpolated_weights = np.empty(
            (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
            dtype=config.dtype_double,
        )
        for state_i in range(tdata.n_states):
            interpolator = self._get_interpolator(
                f"{FV.WEIGHT}:{state_i}",
                support,
                weights[state_i],
                axes,
            )
            query = points[state_i].reshape(-1, 3)
            if axes is not None and same_height:
                query = query[:, :2]
            interpolated_weights[state_i] = interpolator(query).reshape(
                tdata.n_targets, tdata.n_tpoints
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