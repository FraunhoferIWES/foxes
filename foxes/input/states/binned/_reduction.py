"""Reduce source states into topology-native sparse histogram datasets.

This module contains the implementation shared by :class:`BinnedFieldData`
and :class:`BinnedPointCloudData`. The helper is composed into those classes;
it is not a states model and performs no runtime target interpolation.

Reduction happens during ``load_data``. Source states are evaluated on the
configured support, grouped into Cartesian histogram bins independently at
each support point, and stored as a canonical :class:`xarray.Dataset`. Its
leading ``state`` coordinate contains retained flat histogram-bin indices.
Statistics and weights remain spatially resolved so the owning native states
class can apply ordinary FOXES chunking and interpolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import QhullError

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config, get_input_path
from foxes.core import States, Turbine, WindFarm, run_with_engine
from foxes.utils import uv2wd, wd2uv, write_nc

from ._output import wind_rose_dataset

if TYPE_CHECKING:
    from foxes.core import Algorithm, LoadedData, Model
    from foxes.input.states.dataset_states import DatasetStates


class _BinnedStateReduction:
    """
    Prepare binned datasets for native field or point-cloud states.

    The helper accepts either source states that still need reduction or an
    existing binned artifact. It owns histogram definitions, sparse-bin
    metadata, artifact validation, missing-value handling, and optional
    artifact output. The public owner remains solely responsible for loading
    the resulting dataset through ``FieldData`` or ``PointCloudData``.

    Canonical field variables have dimensions ``(state, x, y, height)``;
    point-cloud variables have dimensions ``(state, point)``. In both cases,
    ``state`` labels are flat Cartesian-bin indices and ``weight`` gives the
    source-state mass assigned to each bin and support location.
    """

    def __init__(
        self,
        states: States | str | Path | xr.Dataset,
        *,
        topology: Literal["field", "point_cloud"],
        bin_vars: Mapping[str, Sequence[float] | int] | None,
        mean_vars: Sequence[str] | None,
        support_points: np.ndarray | None,
        support_grid: Mapping[str, Sequence[float]] | None,
        output_file: str | Path | None,
        interpolation: str,
        nan_policy: Literal["raise", "interpolate", "remove"],
        nan_threshold: float,
    ) -> None:
        """
        Initialize the reduction configuration.

        Parameters
        ----------
        states
            Source states to reduce, or a path or dataset containing a
            canonical binned artifact.
        topology
            Native output topology: ``"field"`` or ``"point_cloud"``.
        bin_vars
            Histogram variables and bin edges. Integer ``WS`` and ``WD``
            definitions are expanded to standard equal-width edges.
        mean_vars
            Additional variables retained as conditional weighted means.
        support_points
            Scattered ``(x, y, height)`` coordinates for point-cloud output.
        support_grid
            Regular ``x``, ``y``, and ``height`` axes for field output.
        output_file
            Optional destination for the canonical NetCDF artifact.
        interpolation
            Spatial interpolation method used to fill missing statistics.
        nan_policy
            Active-cell policy: ``"raise"``, ``"interpolate"``, or
            ``"remove"``. Removal is available only for point clouds.
        nan_threshold
            Maximum active-bin missing fraction retained by ``"remove"``.
        """
        self.states = states if isinstance(states, States) else None
        self.input_source: str | Path | xr.Dataset | None = (
            None if self.states is not None else cast(str | Path | xr.Dataset, states)
        )
        self.topology = topology
        self.bin_vars: dict[str, np.ndarray] = {}
        if bin_vars is None and self.states is not None:
            raise ValueError(f"{self.class_name}: Require bin_vars for source states")
        if bin_vars is not None:
            self._set_bin_vars(bin_vars)
        self.mean_vars = None if mean_vars is None else list(mean_vars)
        self._check_mean_vars()
        self.support_points = (
            None
            if support_points is None
            else np.asarray(support_points, dtype=config.dtype_double)
        )
        self.support_grid = (
            None
            if support_grid is None
            else {
                var: np.asarray(values, dtype=config.dtype_double)
                for var, values in support_grid.items()
            }
        )
        self.output_file = None if output_file is None else Path(output_file)
        self.interpolation = interpolation
        if nan_policy not in {"raise", "interpolate", "remove"}:
            raise ValueError(
                f"{self.class_name}: nan_policy must be 'raise', 'interpolate', or 'remove'"
            )
        if topology == "field" and nan_policy == "remove":
            raise ValueError(
                "BinnedFieldData: nan_policy='remove' is incompatible with regular-grid topology"
            )
        if not 0.0 <= nan_threshold <= 1.0:
            raise ValueError(
                f"{self.class_name}: nan_threshold must be between 0 and 1"
            )
        self.nan_policy = nan_policy
        self.nan_threshold = nan_threshold
        self._bin_shape = tuple(len(edges) - 1 for edges in self.bin_vars.values())
        self._n_bins = int(np.prod(self._bin_shape, dtype=np.int64))
        self._state_indices = np.arange(self._n_bins, dtype=config.dtype_int)
        self._calculation_vars: list[str] | None = None
        self._validate_support()

    @property
    def class_name(self) -> str:
        """
        Return the public states class name for this topology.

        Returns
        -------
        str
            ``"BinnedFieldData"`` or ``"BinnedPointCloudData"``.
        """
        return "BinnedFieldData" if self.topology == "field" else "BinnedPointCloudData"

    def _validate_support(self) -> None:
        """
        Validate support configuration for the selected topology.

        Source-backed field data requires one regular grid, whereas
        source-backed point-cloud data requires one coordinate array.
        Artifact-backed objects must not receive constructor support because
        the artifact owns its topology.

        Raises
        ------
        ValueError
            If support is missing, supplied for the wrong topology, or has an
            invalid shape or coordinate ordering.
        """
        support = self.support_grid if self.topology == "field" else self.support_points
        other = self.support_points if self.topology == "field" else self.support_grid
        if self.input_source is not None:
            if support is not None or other is not None:
                raise ValueError(
                    f"{self.class_name}: Input artifacts provide their own support topology"
                )
            return
        if support is None or other is not None:
            name = "support_grid" if self.topology == "field" else "support_points"
            raise ValueError(f"{self.class_name}: Require exactly one {name}")
        if self.topology == "field":
            assert self.support_grid is not None
            if set(self.support_grid) != {FV.X, FV.Y, FV.H}:
                raise ValueError(
                    f"BinnedFieldData: support_grid must contain {FV.X}, {FV.Y}, and {FV.H}"
                )
            if any(
                values.ndim != 1
                or len(values) == 0
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0.0)
                for values in self.support_grid.values()
            ):
                raise ValueError(
                    "BinnedFieldData: Grid axes must be finite, non-empty, increasing one-dimensional arrays"
                )
        else:
            assert self.support_points is not None
            if self.support_points.ndim != 2 or self.support_points.shape[1] != 3:
                raise ValueError(
                    "BinnedPointCloudData: support_points must have shape (n_points, 3)"
                )
            if len(self.support_points) == 0:
                raise ValueError(
                    "BinnedPointCloudData: support_points must not be empty"
                )
            if not np.all(np.isfinite(self.support_points)):
                raise ValueError(
                    "BinnedPointCloudData: support_points must contain only finite coordinates"
                )

    def _set_bin_vars(
        self,
        bin_vars: Mapping[str, Sequence[float] | int],
    ) -> None:
        """
        Validate and store histogram definitions.

        Parameters
        ----------
        bin_vars
            Variables mapped to increasing edges or, for ``WS`` and ``WD``,
            a positive number of standard equal-width bins.
        """
        self.bin_vars = {}
        for var, edges in bin_vars.items():
            if isinstance(edges, (int, np.integer)) and not isinstance(edges, bool):
                if edges < 1:
                    raise ValueError(
                        f"{self.class_name}: Integer bin count must be positive"
                    )
                if var == FV.WS:
                    edges = np.linspace(0.0, 30.0, edges + 1)
                elif var == FV.WD:
                    width = 360.0 / edges
                    edges = np.arange(edges + 1) * width - width / 2.0
                else:
                    raise ValueError(
                        f"{self.class_name}: Integer bins require '{FV.WS}' or '{FV.WD}'"
                    )
            values = np.asarray(edges, dtype=config.dtype_double)
            if (
                values.ndim != 1
                or len(values) < 2
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0.0)
            ):
                raise ValueError(
                    f"{self.class_name}: Invalid increasing bin edges for '{var}'"
                )
            self.bin_vars[var] = values
        if not self.bin_vars:
            raise ValueError(f"{self.class_name}: Require at least one binned variable")
        self._bin_shape = tuple(len(edges) - 1 for edges in self.bin_vars.values())
        self._n_bins = int(np.prod(self._bin_shape, dtype=np.int64))
        self._state_indices = np.arange(self._n_bins, dtype=config.dtype_int)

    def _check_mean_vars(self) -> None:
        """
        Check that histogram and conditional-mean variables are disjoint.

        Raises
        ------
        ValueError
            If a variable occurs in both ``bin_vars`` and ``mean_vars``.
        """
        if self.mean_vars is None:
            return
        duplicates = set(self.bin_vars).intersection(self.mean_vars)
        if duplicates:
            raise ValueError(
                f"{self.class_name}: mean_vars must not duplicate bin_vars: {sorted(duplicates)}"
            )

    def sub_models(self) -> list[Model]:
        """
        Return the wrapped source states model, if present.

        Returns
        -------
        list[foxes.core.Model]
            One source model for reduction, or an empty artifact-input list.
        """
        return [] if self.states is None else [self.states]

    def size(self) -> int:
        """
        Return the current number of histogram states.

        Returns
        -------
        int
            Full Cartesian bin count before preparation, then retained
            non-empty bin count after preparation.
        """
        return self._n_bins

    def index(self) -> list[int]:
        """
        Return the retained flat histogram-bin indices.

        Returns
        -------
        list[int]
            Bin indices used as native ``FC.STATE`` labels.
        """
        return self._state_indices.tolist()

    def output_vars(self, algo: Algorithm) -> list[str]:
        """
        Resolve and return public calculation variables.

        Parameters
        ----------
        algo
            Algorithm used to query source-state output variables.

        Returns
        -------
        list[str]
            Histogram variables followed by conditional-mean variables.
        """
        if self.states is not None and self.mean_vars is None:
            self.mean_vars = [
                var
                for var in self.states.output_point_vars(algo)
                if var not in self.bin_vars
            ]
        return [*self.bin_vars, *(self.mean_vars or [])]

    def calculation_vars(self, algo: Algorithm) -> list[str]:
        """
        Return and cache variables stored for native interpolation.

        Parameters
        ----------
        algo
            Algorithm used while resolving default mean variables.

        Returns
        -------
        list[str]
            Independent copy of the native calculation-variable list.
        """
        if self._calculation_vars is not None:
            return self._calculation_vars.copy()
        self._calculation_vars = self.output_vars(algo)
        return self._calculation_vars.copy()

    def _materialize_support(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Materialize support coordinates for source evaluation.

        Returns
        -------
        support
            Coordinates with shape ``(n_support, 3)``.
        axes
            Regular ``x``, ``y``, and ``height`` axes for field topology, or
            ``None`` for point-cloud topology.
        """
        if self.topology == "point_cloud":
            assert self.support_points is not None
            return self.support_points.copy(), None
        assert self.support_grid is not None
        axes = tuple(self.support_grid[var] for var in (FV.X, FV.Y, FV.H))
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([values.ravel() for values in mesh], axis=-1), axes

    @staticmethod
    def _source_dataset(loaded_data: LoadedData) -> xr.Dataset:
        """
        Collect state-dependent source data from shared model data.

        Parameters
        ----------
        loaded_data
            Data populated while initializing the wrapped source states.

        Returns
        -------
        xarray.Dataset
            Source variables whose dimensions contain ``FC.STATE``.
        """
        data_vars = {
            name: (dims, data)
            for name, (dims, data) in loaded_data["data_vars"].items()
            if FC.STATE in dims
        }
        coords = {
            name: data
            for name, data in loaded_data["coords"].items()
            if isinstance(data, np.ndarray)
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def _read_input_dataset(self) -> xr.Dataset:
        """
        Read the configured artifact into memory.

        Returns
        -------
        xarray.Dataset
            Deep copy of dataset input or fully loaded file contents.
        """
        assert self.input_source is not None
        if isinstance(self.input_source, xr.Dataset):
            return self.input_source.copy(deep=True)
        input_path = cast(str | Path, self.input_source)
        with xr.open_dataset(
            get_input_path(input_path), engine=config.nc_engine
        ) as data:
            return data.load()

    @staticmethod
    def _attr_names(data: xr.Dataset, name: str) -> list[str]:
        """
        Read a comma-separated dataset attribute as a list of names.

        Parameters
        ----------
        data
            Dataset containing the artifact metadata.
        name
            Attribute name.

        Returns
        -------
        list[str]
            Stored names, or an empty list when the attribute is empty or
            absent.
        """
        value = str(data.attrs.get(name, ""))
        return [] if not value else value.split(",")

    def _configure_from_artifact(self, data: xr.Dataset) -> None:
        """
        Validate artifact identity and resolve its stored configuration.

        Parameters
        ----------
        data
            Loaded canonical artifact for this public topology class.
        """
        found_class = data.attrs.get("foxes_state_class")
        if found_class != self.class_name:
            raise ValueError(
                f"{self.class_name}: Expected artifact class '{self.class_name}', got '{found_class}'"
            )
        names = self._attr_names(data, "foxes_binned_bin_vars")
        artifact_bins = {
            name: np.asarray(data.attrs[f"{name}_bounds"]) for name in names
        }
        if self.bin_vars:
            if list(self.bin_vars) != names or any(
                not np.array_equal(self.bin_vars[name], artifact_bins[name])
                for name in names
            ):
                raise ValueError(
                    f"{self.class_name}: Constructor bin_vars differ from artifact bins"
                )
        else:
            self._set_bin_vars(artifact_bins)
        stored_means = self._attr_names(data, "foxes_binned_mean_vars")
        if self.mean_vars is None:
            self.mean_vars = stored_means
        else:
            missing = set(self.mean_vars).difference(stored_means)
            if missing:
                raise KeyError(
                    f"{self.class_name}: Mean variables missing from artifact: {sorted(missing)}"
                )
        stored_calc = self._attr_names(data, "foxes_binned_calculation_vars")
        public = self.output_vars(cast("Algorithm", None))
        missing = set(public).difference(stored_calc)
        if missing:
            raise KeyError(
                f"{self.class_name}: Calculation variables missing from artifact: {sorted(missing)}"
            )
        self._calculation_vars = public

    @staticmethod
    def _stat_var(var: str, stat: str) -> str:
        """
        Return an artifact statistic variable name.

        Parameters
        ----------
        var
            FOXES variable name.
        stat
            Statistic suffix such as ``"min"``, ``"mean"``, or ``"max"``.

        Returns
        -------
        str
            Name in ``<variable>_<statistic>`` form.
        """
        return f"{var}_{stat}"

    def _bin_centres(self) -> np.ndarray:
        """
        Return centers of all Cartesian histogram bins.

        Returns
        -------
        numpy.ndarray
            Centers with shape ``(n_bins, n_bin_vars)``. Wind direction is
            wrapped to ``[0, 360)``.
        """
        centres = []
        for var, edges in self.bin_vars.items():
            values = 0.5 * (edges[:-1] + edges[1:])
            centres.append(np.mod(values, 360.0) if var == FV.WD else values)
        mesh = np.meshgrid(*centres, indexing="ij")
        return np.stack([values.ravel() for values in mesh], axis=-1)

    @staticmethod
    def _circular_bin_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Shift circular values into the interval beginning at ``edges[0]``.

        Parameters
        ----------
        values
            Wind directions in degrees.
        edges
            Wind-direction bin edges spanning one 360-degree interval.

        Returns
        -------
        numpy.ndarray
            Directions shifted into ``[edges[0], edges[0] + 360)``.
        """
        return edges[0] + np.mod(values - edges[0], 360.0)

    def _support_from_artifact(
        self, data: xr.Dataset
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None, tuple[str, ...]]:
        """
        Read topology-native support coordinates from an artifact.

        Parameters
        ----------
        data
            Canonical field or point-cloud artifact.

        Returns
        -------
        support
            Flattened ``(x, y, height)`` coordinates.
        axes
            Regular field axes, or ``None`` for a point cloud.
        support_dims
            Native support dimensions used by artifact variables.
        """
        if self.topology == "field":
            missing = [var for var in (FV.X, FV.Y, FV.H) if var not in data.coords]
            if missing:
                raise KeyError(
                    f"BinnedFieldData: Missing support coordinates {missing}"
                )
            axes = tuple(np.asarray(data[var]) for var in (FV.X, FV.Y, FV.H))
            if any(
                len(axis) == 0
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0.0)
                for axis in axes
            ):
                raise ValueError(
                    "BinnedFieldData: Artifact grid axes must be finite, non-empty, and increasing"
                )
            mesh = np.meshgrid(*axes, indexing="ij")
            support = np.stack([values.ravel() for values in mesh], axis=-1)
            return support, axes, (FV.X, FV.Y, FV.H)
        missing = [var for var in (FV.X, FV.Y, FV.H) if var not in data]
        if missing:
            raise KeyError(f"BinnedPointCloudData: Missing support variables {missing}")
        support = np.stack(
            [np.asarray(data[var]) for var in (FV.X, FV.Y, FV.H)], axis=-1
        )
        if not np.all(np.isfinite(support)):
            raise ValueError(
                "BinnedPointCloudData: Artifact support must contain only finite coordinates"
            )
        return support, None, (FC.POINT,)

    @staticmethod
    def _invalid_stat_cells(
        stats: dict[str, dict[str, np.ndarray]], weights: np.ndarray
    ) -> np.ndarray:
        """
        Find non-finite statistics in active bin-support cells.

        Parameters
        ----------
        stats
            Nested mapping from output variable and statistic to arrays with
            shape ``(n_bins, n_support)``.
        weights
            Bin weights with shape ``(n_bins, n_support)``.

        Returns
        -------
        numpy.ndarray
            Boolean mask with the same shape as ``weights``. A cell is true
            when it has non-zero weight and any statistic is non-finite.
        """
        invalid = np.zeros_like(weights, dtype=bool)
        for values_by_stat in stats.values():
            for values in values_by_stat.values():
                invalid |= ~np.isfinite(values)
        return invalid & (weights != 0.0)

    def _raise_invalid_stats(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
        support: np.ndarray,
    ) -> None:
        """
        Reject the first non-finite statistic in an active cell.

        Parameters
        ----------
        stats
            Nested statistic arrays with shape ``(n_bins, n_support)``.
        weights
            Bin weights defining active cells by non-zero values.
        support
            Support coordinates with shape ``(n_support, 3)``.

        Raises
        ------
        ValueError
            If an active cell contains a non-finite statistic.
        """
        for var, values_by_stat in stats.items():
            for stat, values in values_by_stat.items():
                invalid = np.argwhere((weights != 0.0) & ~np.isfinite(values))
                if len(invalid):
                    bin_i, point_i = invalid[0]
                    raise ValueError(
                        f"{self.class_name}: Non-finite {var}_{stat} for active bin "
                        f"{bin_i} at support point {support[point_i]}"
                    )

    def _fill_invalid_values(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        missing: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate selected missing values with nearest-neighbor fallback.

        Parameters
        ----------
        coordinates
            Two- or three-dimensional support coordinates.
        values
            Values at every support coordinate.
        missing
            Boolean mask selecting values to fill.

        Returns
        -------
        numpy.ndarray
            Filled values for the selected locations, in mask order.
        """
        valid = ~missing
        try:
            filled = griddata(
                coordinates[valid],
                values[valid],
                coordinates[missing],
                method=self.interpolation,
                fill_value=np.nan,
            )
        except (QhullError, ValueError):
            filled = np.full(np.count_nonzero(missing), np.nan)
        unresolved = ~np.isfinite(filled)
        if np.any(unresolved):
            filled[unresolved] = griddata(
                coordinates[valid],
                values[valid],
                coordinates[missing][unresolved],
                method="nearest",
            )
        return filled

    def _interpolate_invalid_stats(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        active: np.ndarray,
        support: np.ndarray,
    ) -> None:
        """
        Interpolate invalid statistics in every retained bin.

        Parameters
        ----------
        stats
            Nested statistic arrays, modified in place.
        active
            Boolean mask selecting bins with non-zero weight somewhere on the
            support.
        support
            Support coordinates with shape ``(n_support, 3)``. Height is
            omitted from interpolation when all support heights are equal.
        """
        same_height = np.allclose(support[:, 2], support[0, 2])
        coordinates = support[:, :2] if same_height else support
        for values_by_stat in stats.values():
            for values in values_by_stat.values():
                for bin_i in np.flatnonzero(active):
                    missing = ~np.isfinite(values[bin_i])
                    if np.any(missing) and np.any(~missing):
                        values[bin_i, missing] = self._fill_invalid_values(
                            coordinates, values[bin_i], missing
                        )

    def _fill_zero_weight_stats(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
        support: np.ndarray,
    ) -> None:
        """
        Complete zero-weight cells for bins retained elsewhere on the support.

        Native spatial interpolation requires finite values across the full
        support for each retained state. Statistics at zero-weight cells do
        not affect probability mass, so they are spatially filled in place.

        Parameters
        ----------
        stats
            Nested statistic arrays, modified in place.
        weights
            Bin weights with shape ``(n_bins, n_support)``.
        support
            Support coordinates with shape ``(n_support, 3)``.
        """
        active = np.any(weights != 0.0, axis=1)
        same_height = np.allclose(support[:, 2], support[0, 2])
        coordinates = support[:, :2] if same_height else support
        for values_by_stat in stats.values():
            for values in values_by_stat.values():
                for bin_i in np.flatnonzero(active):
                    missing = (weights[bin_i] == 0.0) & ~np.isfinite(values[bin_i])
                    if np.any(missing):
                        values[bin_i, missing] = self._fill_invalid_values(
                            coordinates, values[bin_i], missing
                        )

    def _apply_nan_policy(
        self,
        support: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None, np.ndarray]:
        """
        Apply the configured policy to non-finite active statistics.

        ``"raise"`` rejects invalid active cells, ``"interpolate"`` fills
        them spatially, and ``"remove"`` first removes point-cloud support
        locations above ``nan_threshold`` before filling remaining gaps.

        Parameters
        ----------
        support
            Support coordinates with shape ``(n_support, 3)``.
        axes
            Regular field axes, or ``None`` for point-cloud support.
        stats
            Nested statistic arrays, modified in place and subset when points
            are removed.
        weights
            Bin weights with shape ``(n_bins, n_support)``.

        Returns
        -------
        support
            Retained support coordinates.
        axes
            Original field axes or ``None`` for point-cloud output.
        weights
            Weights restricted to retained support locations.

        Raises
        ------
        ValueError
            If invalid statistics remain or removal discards every support
            point.
        """
        active = np.any(weights != 0.0, axis=1)
        self._fill_zero_weight_stats(stats, weights, support)
        invalid = self._invalid_stat_cells(stats, weights)
        if not np.any(invalid):
            return support, axes, weights
        if self.nan_policy == "raise":
            self._raise_invalid_stats(stats, weights, support)
        elif self.nan_policy == "remove":
            missing_fraction = np.mean(invalid[active], axis=0)
            keep = missing_fraction <= self.nan_threshold
            if not np.any(keep):
                raise ValueError(
                    "BinnedPointCloudData: nan_policy='remove' removed all support points"
                )
            support = support[keep]
            weights = weights[:, keep]
            for values_by_stat in stats.values():
                for stat, values in values_by_stat.items():
                    values_by_stat[stat] = values[:, keep]
            axes = None
        self._interpolate_invalid_stats(stats, active, support)
        self._raise_invalid_stats(stats, weights, support)
        return support, axes, weights

    def _create_output_dataset(
        self,
        support: np.ndarray,
        axes: tuple[np.ndarray, ...] | None,
        stats: dict[str, dict[str, np.ndarray]],
        weights: np.ndarray,
        bin_indices: np.ndarray | None = None,
        bin_centres: np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Create a canonical topology-native sparse dataset.

        Empty bins are discarded. Retained flat Cartesian bin indices become
        the ``FC.STATE`` coordinate. Runtime variables contain conditional
        means, except binned wind direction, which contains its bin center.

        Parameters
        ----------
        support
            Flattened support coordinates with shape ``(n_support, 3)``.
        axes
            Regular field axes, or ``None`` for point-cloud output.
        stats
            Nested statistic arrays with shape ``(n_bins, n_support)``.
        weights
            Bin weights with shape ``(n_bins, n_support)``.
        bin_indices
            Optional flat Cartesian labels for the input rows.
        bin_centres
            Optional centers corresponding to the input rows.

        Returns
        -------
        xarray.Dataset
            Canonical field or point-cloud artifact containing statistics,
            runtime variables, spatial weights, and sparse state metadata.

        Raises
        ------
        ValueError
            If support coordinates, weights, retained bin centres, or active
            statistics are non-finite.
        """
        if not np.all(np.isfinite(support)):
            raise ValueError(f"{self.class_name}: Non-finite support coordinates")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{self.class_name}: Non-finite weights")
        self._fill_zero_weight_stats(stats, weights, support)
        self._raise_invalid_stats(stats, weights, support)
        active = np.any(weights != 0.0, axis=1)
        if bin_indices is None:
            bin_indices = np.arange(len(weights), dtype=config.dtype_int)
        bin_indices = np.asarray(bin_indices)[active]
        if bin_centres is None:
            bin_centres = self._bin_centres()[bin_indices]
        else:
            bin_centres = np.asarray(bin_centres)[active]
        if not np.all(np.isfinite(bin_centres)):
            raise ValueError(f"{self.class_name}: Non-finite bin centres")
        weights = weights[active]
        filtered_stats = {
            var: {stat: values[active] for stat, values in values_by_stat.items()}
            for var, values_by_stat in stats.items()
        }
        self._n_bins = len(bin_indices)
        self._state_indices = np.asarray(bin_indices, dtype=config.dtype_int)
        attrs: dict[str, Any] = {
            "foxes_state_class": self.class_name,
            "foxes_binned_bin_vars": ",".join(self.bin_vars),
            "foxes_binned_mean_vars": ",".join(self.mean_vars or []),
            "foxes_binned_calculation_vars": ",".join(self._calculation_vars or []),
        }
        attrs.update({f"{var}_bounds": edges for var, edges in self.bin_vars.items()})
        if config.utm_zone_set:
            number, letter = config.utm_zone
            attrs["utm_zone"] = f"{number}{letter}"
        state_dims = (FC.STATE, "binned_state_var")
        support_dims: tuple[str, ...]
        if self.topology == "field":
            assert axes is not None
            support_dims = (FV.X, FV.Y, FV.H)
            support_shape = tuple(len(axis) for axis in axes)
            coords: dict[str, Any] = {
                FV.X: axes[0],
                FV.Y: axes[1],
                FV.H: axes[2],
            }
            support_vars: dict[str, Any] = {}
        else:
            support_dims = (FC.POINT,)
            support_shape = (len(support),)
            coords = {
                FC.POINT: np.arange(len(support), dtype=config.dtype_int),
                FC.XYH: (FC.XYH, np.asarray([FV.X, FV.Y, FV.H])),
                "support": ((FC.POINT, FC.XYH), support),
            }
            support_vars = {
                FV.X: ((FC.POINT,), support[:, 0]),
                FV.Y: ((FC.POINT,), support[:, 1]),
                FV.H: ((FC.POINT,), support[:, 2]),
            }
        data_vars = {
            self._stat_var(var, stat): (
                (FC.STATE,) + support_dims,
                values.reshape((self._n_bins,) + support_shape),
            )
            for var, values_by_stat in filtered_stats.items()
            for stat, values in values_by_stat.items()
        }
        for var in self._calculation_vars or []:
            if var == FV.WD and var in self.bin_vars:
                var_i = list(self.bin_vars).index(var)
                values = np.broadcast_to(
                    bin_centres[:, var_i, None],
                    (self._n_bins, len(support)),
                )
            else:
                values = filtered_stats[var]["mean"]
            data_vars[var] = (
                (FC.STATE,) + support_dims,
                values.reshape((self._n_bins,) + support_shape),
            )
        data_vars[FV.WEIGHT] = (
            (FC.STATE,) + support_dims,
            weights.reshape((self._n_bins,) + support_shape),
        )
        data_vars.update(support_vars)
        return xr.Dataset(
            data_vars=data_vars,
            coords={
                FC.STATE: self._state_indices,
                "binned_state_var": list(self.bin_vars),
                "bin_centres": (state_dims, bin_centres),
                **coords,
            },
            attrs=attrs,
        )

    def _prepare_artifact(self) -> xr.Dataset:
        """
        Validate and normalize the configured input artifact.

        Stored topology, bins, variables, coordinates, weights, statistics,
        sparse state labels, and bin centers are checked before the configured
        missing-value policy is applied.

        Returns
        -------
        xarray.Dataset
            Canonical in-memory artifact ready for the native states loader.

        Raises
        ------
        KeyError
            If required coordinates, metadata, weights, or statistics are
            missing.
        ValueError
            If topology, bin definitions, weights, or statistics are invalid.
        """
        data = self._read_input_dataset()
        self._configure_from_artifact(data)
        support, axes, support_dims = self._support_from_artifact(data)
        full_dims = (FC.STATE,) + support_dims
        n_states = data.sizes[FC.STATE]
        n_support = len(support)
        if FV.WEIGHT not in data:
            raise KeyError(f"{self.class_name}: Missing '{FV.WEIGHT}'")
        weights = data[FV.WEIGHT].transpose(*full_dims).to_numpy()
        weights = weights.reshape(n_states, n_support)
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{self.class_name}: Non-finite artifact weights")
        stats: dict[str, dict[str, np.ndarray]] = {}
        for var in self._calculation_vars or []:
            stat_names = ("min", "mean", "max") if var in self.bin_vars else ("mean",)
            stats[var] = {}
            for stat in stat_names:
                name = self._stat_var(var, stat)
                if name not in data:
                    raise KeyError(f"{self.class_name}: Missing '{name}'")
                values = data[name].transpose(*full_dims).to_numpy()
                stats[var][stat] = values.reshape(n_states, n_support)
        support, axes, weights = self._apply_nan_policy(support, axes, stats, weights)
        return self._create_output_dataset(
            support,
            axes,
            stats,
            weights,
            bin_indices=np.asarray(data[FC.STATE]),
            bin_centres=np.asarray(data["bin_centres"]),
        )

    def _reduce_source(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        verbosity: int,
    ) -> xr.Dataset:
        """
        Evaluate source states and reduce them into spatial histogram bins.

        The wrapped states are calculated at every support point without wake
        effects. Source-state weights populate a separate histogram at each
        support location. Wind-direction means use vector averaging.

        Parameters
        ----------
        algo
            Owning calculation algorithm, used to position the temporary
            no-wake farm and resolve source outputs.
        loaded_data
            Shared data containing initialized source-state coordinates and
            optional state-only weights.
        verbosity
            Verbosity forwarded to the temporary source algorithm.

        Returns
        -------
        xarray.Dataset
            Canonical sparse binned artifact.

        Raises
        ------
        KeyError
            If the source calculation omits a requested variable.
        ValueError
            If source weights or calculated variable shapes violate the
            reduction contract, or active statistics remain non-finite.
        """
        support, axes = self._materialize_support()
        assert self.states is not None
        calculation_vars = self.calculation_vars(algo)
        source_data = self._source_dataset(loaded_data)
        from foxes.algorithms import Downwind

        farm = WindFarm()
        farm.add_turbine(
            Turbine(
                xy=algo.farm.turbines[0].xy,
                turbine_models=["null_type"],
                H=support[0, 2],
            ),
            verbosity=0,
        )
        source_algo = Downwind(
            farm=farm,
            states=self.states,
            rotor_model="centre",
            partial_wakes="centre",
            wake_models=[],
            verbosity=verbosity - 1,
        )
        source_algo.initialize(force=True)

        def _calc_source() -> xr.Dataset:
            farm_results = source_algo.calc_farm(ambient=True)
            results = source_algo.calc_points(
                farm_results,
                support,
                outputs=[FV.var2amb.get(var, var) for var in calculation_vars],
                ambient=True,
            )
            return results.rename(
                {var: FV.amb2var[var] for var in results if var in FV.amb2var}
            )

        source_results = run_with_engine(_calc_source)
        n_source_states = self.states.size()
        if FV.WEIGHT in source_data:
            if source_data[FV.WEIGHT].dims != (FC.STATE,):
                raise ValueError(
                    f"{self.class_name}: Source weights must be state-dependent only"
                )
            source_weights = np.asarray(source_data[FV.WEIGHT])
        else:
            source_weights = np.full(
                n_source_states,
                1.0 / n_source_states,
                dtype=config.dtype_double,
            )
        if source_weights.shape != (n_source_states,):
            raise ValueError(
                f"{self.class_name}: Source weights require one value per source state"
            )
        if not np.all(np.isfinite(source_weights)):
            raise ValueError(f"{self.class_name}: Source weights must be finite")

        bin_indices = []
        source_values: dict[str, np.ndarray] = {}
        for var in calculation_vars:
            if var not in source_results:
                raise KeyError(
                    f"{self.class_name}: Source does not provide variable '{var}'"
                )
            values = source_results[var].to_numpy()
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.shape != (n_source_states, len(support)):
                raise ValueError(
                    f"{self.class_name}: Source variable '{var}' requires shape "
                    f"{(n_source_states, len(support))}, got {values.shape}"
                )
            source_values[var] = values
        for var, edges in self.bin_vars.items():
            values = source_values[var]
            if var == FV.WD:
                values = self._circular_bin_values(values, edges)
            bin_indices.append(np.searchsorted(edges, values, side="right") - 1)

        valid = np.ones((n_source_states, len(support)), dtype=bool)
        flat_bin = np.zeros_like(valid, dtype=config.dtype_int)
        multiplier = 1
        for indices, n_bins in zip(reversed(bin_indices), reversed(self._bin_shape)):
            valid &= (indices >= 0) & (indices < n_bins)
            flat_bin += np.where(valid, indices, 0) * multiplier
            multiplier *= n_bins

        weights = np.zeros((self._n_bins, len(support)), dtype=config.dtype_double)
        for point_i in range(len(support)):
            mask = valid[:, point_i]
            np.add.at(
                weights[:, point_i],
                flat_bin[mask, point_i],
                source_weights[mask],
            )

        stats: dict[str, dict[str, np.ndarray]] = {}
        for var in calculation_vars:
            values = source_values[var]
            is_bin_var = var in self.bin_vars
            mean = np.zeros_like(weights)
            if is_bin_var:
                minimum = np.full_like(weights, np.inf)
                maximum = np.full_like(weights, -np.inf)
            if var == FV.WD:
                vectors = np.zeros(weights.shape + (2,), dtype=config.dtype_double)
                for point_i in range(len(support)):
                    mask = valid[:, point_i]
                    bins = flat_bin[mask, point_i]
                    point_weights = source_weights[mask]
                    np.add.at(
                        vectors[:, point_i],
                        bins,
                        wd2uv(values[mask, point_i], point_weights),
                    )
                    if is_bin_var:
                        circular = self._circular_bin_values(
                            values[mask, point_i], self.bin_vars[var]
                        )
                        np.minimum.at(minimum[:, point_i], bins, circular)
                        np.maximum.at(maximum[:, point_i], bins, circular)
                mean[:] = uv2wd(vectors)
                mean[np.isclose(mean, 360.0)] = 0.0
            else:
                for point_i in range(len(support)):
                    mask = valid[:, point_i]
                    bins = flat_bin[mask, point_i]
                    np.add.at(
                        mean[:, point_i],
                        bins,
                        source_weights[mask] * values[mask, point_i],
                    )
                    if is_bin_var:
                        np.minimum.at(minimum[:, point_i], bins, values[mask, point_i])
                        np.maximum.at(maximum[:, point_i], bins, values[mask, point_i])
                np.divide(mean, weights, out=mean, where=weights > 0.0)
            mean[weights <= 0.0] = np.nan
            stats[var] = {"mean": mean}
            if is_bin_var:
                minimum[weights <= 0.0] = np.nan
                maximum[weights <= 0.0] = np.nan
                stats[var] = {
                    "min": minimum,
                    "mean": mean,
                    "max": maximum,
                }

        support, axes, weights = self._apply_nan_policy(support, axes, stats, weights)
        return self._create_output_dataset(support, axes, stats, weights)

    def prepare_dataset(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        verbosity: int,
    ) -> xr.Dataset:
        """
        Prepare and optionally write the canonical reduced dataset.

        Parameters
        ----------
        algo
            Owning calculation algorithm.
        loaded_data
            Shared model data, including wrapped source data when reduction is
            required.
        verbosity
            Verbosity used for source evaluation and artifact writing.

        Returns
        -------
        xarray.Dataset
            Validated artifact from either source reduction or artifact input.
        """
        data = (
            self._prepare_artifact()
            if self.input_source is not None
            else self._reduce_source(algo, loaded_data, verbosity)
        )
        if self.output_file is not None:
            write_nc(data, self.output_file, pack=True, verbosity=verbosity)
        return data

    def preserve_source_data(
        self,
        owner: DatasetStates,
        loaded_data: LoadedData,
    ) -> None:
        """
        Move wrapped source arrays away from the reduced state dimension.

        The source and histogram generally have different state counts. Before
        the native loader installs reduced arrays, this method renames the
        source ``FC.STATE`` dimension to a model-private coordinate.

        Parameters
        ----------
        owner
            Public binned states object providing namespaced variable names.
        loaded_data
            Shared model data modified in place.
        """
        if self.states is None:
            return
        source_state = owner.var(FC.STATE + "0")
        if source_state in loaded_data["coords"]:
            return
        state_coord = loaded_data["coords"].pop(FC.STATE, None)
        if state_coord is not None:
            loaded_data["coords"][source_state] = state_coord
        for name, (dims, values) in list(loaded_data["data_vars"].items()):
            if FC.STATE in dims:
                loaded_data["data_vars"][name] = (
                    tuple(source_state if dim == FC.STATE else dim for dim in dims),
                    values,
                )

    def install_metadata(
        self,
        owner: DatasetStates,
        loaded_data: LoadedData,
        data: xr.Dataset,
    ) -> None:
        """
        Install sparse histogram metadata beside native loaded data.

        Parameters
        ----------
        owner
            Public binned states object providing namespaced metadata keys.
        loaded_data
            Shared model data modified in place.
        data
            Canonical artifact containing retained state labels and centers.
        """
        bin_vars_key = owner.var("bin_vars")
        bin_centres_key = owner.var("bin_centres")
        loaded_data["coords"][bin_vars_key] = np.asarray(list(self.bin_vars))
        loaded_data["data_vars"][bin_centres_key] = (
            (FC.STATE, bin_vars_key),
            np.asarray(data["bin_centres"]),
        )
        loaded_data["extra_data"][owner.var("bin_indices")] = np.asarray(
            data[FC.STATE], dtype=config.dtype_int
        )

    @staticmethod
    def _loaded_variable(
        owner: DatasetStates,
        loaded_data: LoadedData,
        variable: str,
    ) -> np.ndarray:
        """
        Extract one variable from grouped native loaded arrays.

        Parameters
        ----------
        owner
            Public native states object whose metadata describes the groups.
        loaded_data
            Shared model data created during initialization.
        variable
            Variable to extract.

        Returns
        -------
        numpy.ndarray
            Variable values with the group's trailing variable axis removed.

        Raises
        ------
        KeyError
            If no grouped array contains ``variable``.
        """
        metadata = loaded_data["extra_data"][owner.META]
        for data_key in metadata["data_keys"]:
            dims, values = loaded_data["data_vars"][data_key]
            names = np.asarray(loaded_data["coords"][dims[-1]]).tolist()
            if variable in names:
                return np.asarray(values)[..., names.index(variable)]
        raise KeyError(f"States '{owner.name}': Loaded variable '{variable}' not found")

    def support_wind_rose_data(
        self,
        owner: DatasetStates,
        loaded_data: LoadedData,
    ) -> xr.Dataset:
        """
        Build per-support wind-rose frequencies from native loaded weights.

        Parameters
        ----------
        owner
            Public field or point-cloud binned states object.
        loaded_data
            Shared model data created during initialization.

        Returns
        -------
        xarray.Dataset
            Dense percentage frequencies with dimensions
            ``(point, WD, WS)`` and explicit support coordinates.

        Raises
        ------
        KeyError
            If the histogram is not defined by exactly ``WD`` and ``WS``.
        """
        if self.topology == "field":
            axes = [
                np.asarray(loaded_data["coords"][owner.var(var)])
                for var in (FV.X, FV.Y, FV.H)
            ]
            mesh = np.meshgrid(*axes, indexing="ij")
            support = np.stack([values.ravel() for values in mesh], axis=-1)
        else:
            support = owner.get_grid_points(loaded_data=loaded_data)
        weights = self._loaded_variable(owner, loaded_data, FV.WEIGHT)
        bin_indices = np.asarray(
            loaded_data["extra_data"][owner.var("bin_indices")],
            dtype=config.dtype_int,
        )
        return wind_rose_dataset(
            support,
            weights,
            self.bin_vars,
            self._bin_shape,
            bin_indices,
        )
