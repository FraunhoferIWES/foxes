"""Bounded arbitrary-point collection and worker-side weighted means."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import Dict, write_nc

from ._dataset_output import DatasetOutput
from ._units import _set_units
from .point_results import PointResultsOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData
    from foxes.core.model import LoadedData


class PointCalculator(DatasetOutput):
    """Collect configured or supplied points through the ordinary output lifecycle.

    State means transport weighted sums, not full state histories. Farm runs
    evaluate configured locations in bounded point blocks without nested engines.
    """

    def __init__(
        self,
        points: Any = None,
        variables: list[str] | None = None,
        states_mean: bool = False,
        weight_turbine: int = 0,
        to_file: str | None = None,
        write_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure points, fields and optional state reduction.

        Parameters
        ----------
        points
            Coordinate configuration in metres, shaped ``(n_points, 3)`` or
            ``(n_states, n_points, 3)``. ``None`` requires supplied point data.
            Materialized coordinates move to loaded data before execution.
        variables
            Point fields; ``None`` selects wind speed.
        states_mean
            Return normalized weighted means on ``(FC.POINT,)`` instead of
            state-resolved fields. Zero-weight samples do not propagate NaNs.
        weight_turbine
            Original farm turbine position supplying state weights for means.
        to_file
            Optional NetCDF filename, written once on the parent. Coordinates
            are included as ``x``, ``y`` and ``z`` in both files and results.
        write_pars
            NetCDF verbosity defaults to the executing algorithm's level.
            Options for ``foxes.utils.write_nc``.
        kwargs
            Result naming, output paths and common options.
        """
        super().__init__(use_points=True, **kwargs)
        fields = [FV.WS] if variables is None else list(variables)
        if not fields or any(not isinstance(v, str) or not v for v in fields):
            raise ValueError("PointCalculator: variables must contain field names")
        if isinstance(weight_turbine, bool) or not isinstance(weight_turbine, int):
            raise ValueError("PointCalculator: weight_turbine must be an integer")
        self.points = points
        self.fields = tuple(fields)
        self._collector = PointResultsOutput(variables=fields)
        self.states_mean = states_mean
        self.weight_turbine = weight_turbine
        self.to_file = to_file
        self.write_pars = {} if write_pars is None else dict(write_pars)
        self._points_key = self.var("points")

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Validate coordinates and load static or state-dependent point data.

        Parameters
        ----------
        algo
            Algorithm supplying the total state count.
        loaded_data
            Mapping extended with private coordinate arrays, never model buffers.
        force
            Overwrite loaded data.
        verbosity
            Initialization verbosity.

        Raises
        ------
        ValueError
            If coordinate shapes/values or the weight turbine are invalid.
        """
        super().load_data(algo, loaded_data, force, verbosity)
        if (
            self.states_mean
            and not -algo.n_turbines <= self.weight_turbine < algo.n_turbines
        ):
            raise ValueError(f"{self.name}: weight_turbine outside the farm")
        if self.points is None:
            return
        points = np.array(self.points, dtype=float, copy=True)
        if (
            points.ndim not in (2, 3)
            or points.shape[-1] != 3
            or points.shape[-2] == 0
            or not np.all(np.isfinite(points))
            or points.ndim == 3
            and points.shape[0] != algo.n_states
        ):
            raise ValueError(
                f"{self.name}: Expected finite (n_points, 3) or "
                f"(n_states, n_points, 3) coordinates, got {points.shape}"
            )
        points.setflags(write=False)
        if points.ndim == 2:
            loaded_data["extra_data"][self._points_key] = points
        else:
            dimension = self.var(FC.POINT)
            loaded_data["coords"][dimension] = np.arange(points.shape[1])
            loaded_data["coords"].setdefault(FC.XYH, np.arange(3))
            loaded_data["data_vars"][self._points_key] = (
                (FC.STATE, dimension, FC.XYH),
                points,
            )

    def get_calc_points(self, algo: Algorithm) -> np.ndarray | None:
        """Retrieve initialized coordinates without retaining materialized grids.

        Parameters
        ----------
        algo
            Algorithm owning this output's loaded configuration.

        Returns
        -------
        points
            Static/dynamic coordinates, or ``None`` for supplied point data only.
        """
        data = algo.loaded_data
        if self._points_key in data["extra_data"]:
            return data["extra_data"][self._points_key]
        entry = data["data_vars"].get(self._points_key)
        return None if entry is None else entry[1]

    def get_calc_points_from_chunk(self, mdata: MData) -> np.ndarray | None:
        """Retrieve static coordinates or the current dynamic state slab.

        Parameters
        ----------
        mdata
            Chunk-local loaded model data and shared extra data.

        Returns
        -------
        points
            Configured coordinates, or ``None`` for supplied point data only.
        """
        return mdata.get(self._points_key, mdata.extra_data.get(self._points_key))

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Stash raw coordinate configuration before read-only worker execution.

        Parameters
        ----------
        algo
            Executing algorithm.
        data_stash
            Lifecycle-owned parent stash.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["points"] = self.points
            self.points = None

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore coordinate configuration after worker execution.

        Parameters
        ----------
        algo
            Executing algorithm.
        data_stash
            Parent stash containing original constructor coordinates.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.points = data_stash[self.name].pop("points")

    def _collect_points(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData | None
    ) -> dict[str, Any]:
        """Reduce each chunk to point sums when state means are requested.

        Parameters
        ----------
        algo
            Executing algorithm with initialized point dependencies.
        mdata
            State-slab metadata and loaded coordinates.
        fdata
            Farm fields supplying original-order turbine state weights.
        tdata
            Supplied targets, or ``None`` for bounded configured-point evaluation.

        Returns
        -------
        payload
            Ordinary collector payload, or state-free weighted field/coordinate
            sums and one scalar weight sum. No running instance is modified.

        Raises
        ------
        ValueError
            If weights are negative/nonfinite or targets have multiple tpoints.
        """
        payload = super()._collect_points(algo, mdata, fdata, tdata)
        if not self.states_mean:
            return payload
        if len(payload["tpoints"]) != 1:
            raise ValueError(
                f"{self.name}: PointCalculator requires one tpoint per target"
            )
        weights = fdata[FV.WEIGHT]
        if weights.ndim == 2:
            weights = weights[:, self.weight_turbine]
        weights = np.broadcast_to(weights, (mdata.n_states,))
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(
                f"{self.name}: State weights must be finite and nonnegative"
            )
        expanded = weights[:, None, None]
        points = np.broadcast_to(
            payload["points"][:, :, 0], (mdata.n_states, len(payload["targets"]), 3)
        )
        return {
            "targets": payload["targets"],
            "mass": float(weights.sum()),
            "points": np.sum(np.where(expanded > 0, points, 0) * expanded, axis=0),
            "values": {
                v: np.sum(np.where(expanded > 0, values, 0) * expanded, axis=0)[:, 0]
                for v, values in payload["values"].items()
            },
        }

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Assemble raw points or merge state-free weighted sums on the parent.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Total engine state chunks.
        n_chunks_targets
            Total engine target chunks.
        output_results
            Engine-owned mapping; only this output's private payload is consumed.

        Raises
        ------
        ValueError
            If weighted means have no positive total weight.
        """
        if not self.states_mean:
            super().finalize_results(
                algo, n_chunks_states, n_chunks_targets, output_results
            )
            return
        chunks = output_results.pop(self.var("data"), {})
        tiles = []
        for target in range(n_chunks_targets):
            slabs = [chunks[(state, target)] for state in range(n_chunks_states)]
            mass = sum(slab["mass"] for slab in slabs)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(
                    f"{self.name}: State mean requires positive total weight"
                )
            points = sum(slab["points"] for slab in slabs) / mass
            tiles.append(
                Dataset(
                    {
                        **{
                            v: (FC.POINT, sum(s["values"][v] for s in slabs) / mass)
                            for v in self.fields
                        },
                        **{
                            axis: (FC.POINT, points[:, i])
                            for i, axis in enumerate("xyz")
                        },
                    },
                    coords={FC.POINT: slabs[0]["targets"]},
                )
            )
        from xarray import concat

        output_results[self.result_name] = self.process(
            algo, concat(tiles, FC.POINT), output_results
        )

    def process(
        self, algo: Algorithm, data: Dataset, output_results: Dict[str, Any]
    ) -> Dataset:
        """Return point fields with compact coordinates and optionally write once.

        Parameters
        ----------
        algo
            Executing algorithm; no additional science is performed.
        data
            State-resolved point fields or merged weighted means.
        output_results
            Run-owned output results; unused by numerical point outputs.

        Returns
        -------
        points
            Selected fields and ``x/y/z`` coordinates, preserving point order.
            Known scientific fields have ``units`` attributes; coordinates use
            metres, including in state-mean results and written NetCDF files.
        """
        if not self.states_mean:
            points = np.stack([data[v].to_numpy() for v in (FV.X, FV.Y, FV.H)], axis=-1)
            dims: tuple[str, ...] = (FC.STATE, FC.POINT)
            if np.array_equal(points, np.broadcast_to(points[:1], points.shape)):
                points = points[0]
                dims = (FC.POINT,)
            data = (
                data[list(self.fields)]
                .drop_vars([v for v in (FV.X, FV.Y, FV.H) if v not in self.fields])
                .assign({axis: (dims, points[..., i]) for i, axis in enumerate("xyz")})
            )

        _set_units(data)
        if self.to_file is not None:
            options = {"verbosity": algo.verbosity, **self.write_pars}
            write_nc(data, self.get_fpath(self.to_file), **options)

        return data
