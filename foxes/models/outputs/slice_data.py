"""Constructor-defined horizontal and vertical slice data outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import Index

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import write_nc

from . import grids
from ._point_data_output import PointDataOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData
    from foxes.core.model import LoadedData
    from foxes.utils import Dict


class SliceData(PointDataOutput):
    """Collect configured point slices and optionally reduce their state axis.

    This numerical output owns slice formatting independently of rendering.
    Flow plots and animations specialize this contract.
    """

    def __init__(
        self,
        plane: str = "xy",
        variables: list[str] | None = None,
        reduction: str = "states",
        data_format: str = "xarray",
        grid_pars: dict[str, Any] | None = None,
        normalize: dict[str, float] | None = None,
        normalize_v: dict[str, float] | None = None,
        label_map: dict[str, str] | None = None,
        vmin: dict[str, float] | None = None,
        vmax: dict[str, float] | None = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] | None = None,
        ret_grid: bool = False,
        ret_states: bool = False,
        result_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure all slice decisions without creating grids or result arrays.

        Parameters
        ----------
        plane
            ``"xy"``, ``"xz"`` or ``"yz"``.
        variables
            Point fields to return; ``None`` selects wind speed.
        reduction
            ``"states"`` retains frames; ``"mean"`` applies state weights.
        data_format
            ``"xarray"``, ``"numpy"`` or ``"pandas"``.
        grid_pars
            Grid resolution/count, bounds, fixed plane coordinate and, for
            vertical planes, ``x_direction`` in degrees.
        normalize
            Coordinate divisors keyed by ``"x"``, ``"y"`` or ``"z"``.
        normalize_v
            Field divisors keyed by variable name.
        label_map
            Names for returned coordinates and fields.
        vmin
            Lower clipping limits keyed by field name.
        vmax
            Upper clipping limits keyed by field name.
        to_file
            Optional destination filename.
        write_pars
            Options for the selected file format.
        ret_grid
            Append static grid metadata to returned data.
        ret_states
            Append retained state labels to returned data.
        result_name
            Mapping key for composed outputs; ``None`` uses this model's name.
        kwargs
            Common output paths, state/chunk selections and ambient mode.
        """
        # Scientific dependencies and grids are deferred to normal initialization.
        fields = [FV.WS] if variables is None else list(variables)
        super().__init__(
            plane=plane,
            variables=variables,
            grid_pars=grid_pars,
            **kwargs,
        )
        self.fields = tuple(fields)
        self.reduction = reduction
        self.data_format = data_format
        self.normalize = {} if normalize is None else dict(normalize)
        self.normalize_v = {} if normalize_v is None else dict(normalize_v)
        self.label_map = {} if label_map is None else dict(label_map)
        self.vmin = {} if vmin is None else dict(vmin)
        self.vmax = {} if vmax is None else dict(vmax)
        self.to_file = to_file
        self.write_pars = {} if write_pars is None else dict(write_pars)
        self.ret_grid = ret_grid
        self.ret_states = ret_states
        self.result_name = result_name or self.name
        self._order_key = self.var("slice_order")

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Validate numerical formatting before initializing point collection.

        Parameters
        ----------
        algo
            Algorithm supplying farm geometry and point models.
        loaded_data
            Shared initialization mapping to extend.
        force
            Reinitialize scientific dependencies and loaded configuration.
        verbosity
            Initialization verbosity.

        Returns
        -------
        loaded_data
            Shared data containing grids, optional selection masks and state ranks.

        Raises
        ------
        ValueError
            If reduction, format, normalization or point configuration is invalid,
            or initialization is attempted while running.
        """
        # Reject unsupported output shapes before loading scientific dependencies.
        if self.reduction not in ("states", "mean"):
            raise ValueError(f"{self.name}: Unknown reduction '{self.reduction}'")
        if self.data_format not in ("numpy", "pandas", "xarray"):
            raise ValueError(f"{self.name}: Unknown data format '{self.data_format}'")
        for value in [*self.normalize.values(), *self.normalize_v.values()]:
            if not np.isfinite(value) or value == 0:
                raise ValueError(
                    f"{self.name}: Normalization divisors must be finite and nonzero"
                )
        return super().initialize(algo, loaded_data, force, verbosity)

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Load grids and state ordering as engine-owned model data.

        Parameters
        ----------
        algo
            Algorithm supplying input state labels and farm geometry.
        loaded_data
            Shared initialization data extended with a static point grid and
            integer ``(FC.STATE,)`` ranks for the requested state order.
        force
            Overwrite previously loaded model data.
        verbosity
            Initialization verbosity.
        """
        super().load_data(algo, loaded_data, force, verbosity)
        states = np.asarray(algo.states.index())
        labels = self.states_sel
        if self.states_isel is not None:
            labels = states[self.states_isel]
        ranks = np.arange(len(states))
        if labels is not None:
            positions = Index(states).get_indexer(labels)
            ranks[positions] = np.arange(len(positions))
        loaded_data["data_vars"][self._order_key] = ((FC.STATE,), ranks)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect numerical targets and their configured state ordering.

        Parameters
        ----------
        algo
            Executing algorithm supplying initialized scientific point models.
        mdata
            Chunk metadata including this output's loaded state ranks and mask.
        fdata
            Farm-ordered scientific inputs.
        tdata
            Supplied target data, or ``None`` to evaluate the configured grid.
        model_results
            Scientific calculation returns passed to the point collector.
        chunk_results
            Shared payload mapping to extend, or ``None`` to create one.

        Returns
        -------
        chunk_results
            Shared mapping extended under this output's private slice-chunk key.
            State-resolved results retain copied fields and state ranks. Means
            contain weighted sums of shape ``(1, n_points)``, static coordinates
            and at most one farm-overlay state, not per-state fields or weights.
            State labels/ranks are retained for means only with ``ret_states``.
        """
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        if self._result_key in results and results[self._result_key] is not None:
            ranks = mdata[self._order_key]
            selection = None if self._selection is None else mdata[self._selection]
            payload = results[self._result_key]
            ranks = ranks if selection is None else ranks[selection]
            if self.reduction == "mean":
                self._reduce_mean_chunk(payload, ranks)
            else:
                payload["order"] = ranks.copy()
        return results

    def _reduce_mean_chunk(self, payload: dict[str, Any], ranks: np.ndarray) -> None:
        """Replace state frames with weighted point sums before worker transport.

        Parameters
        ----------
        payload
            Mutable chunk payload. Fields become ``(1, point)`` weighted sums;
            coordinates and overlays retain only one state row. Raw weights
            and unused state labels are removed without renormalizing weights.
        ranks
            Original input-state positions, used to select the first overlay
            and preserve requested state-label ordering.

        Raises
        ------
        ValueError
            If target coordinates change across states.
        """
        points = payload["points"]
        if not np.allclose(points, points[:1], rtol=0, atol=1e-10):
            raise ValueError(f"{self.name}: Slice targets must be static across states")
        weights = payload["values"][FV.WEIGHT]
        payload["values"] = {
            v: np.sum(payload["values"][v] * weights, axis=0, keepdims=True)
            for v in self.fields
        }
        first = int(np.argmin(ranks))
        payload["farm"] = {
            v: d[first : first + 1].copy() for v, d in payload["farm"].items()
        }
        payload["points"] = points[:1].copy()
        payload["first_rank"] = int(ranks[first])
        payload["order"] = ranks.copy() if self.ret_states else np.empty(0, dtype=int)
        if not self.ret_states:
            payload["states"] = np.array(["mean"])

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Combine point tiles and return the configured slice representation.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total point chunks.
        output_results
            Run-local buffers, replaced by this output's formatted result.
        """
        # Assemble numerical slices in their configured state order.
        chunks = output_results.pop(self._result_key, {})
        selected = self._selected_chunks(chunks, n_chunks_states, n_chunks_targets)
        if not selected:
            raise ValueError(f"{self.name}: No selected slice data")
        payload = self._combine_chunks(selected)
        if self.reduction == "mean":
            output_results[self.result_name] = self.format_payload(algo, payload)
            return
        first_tiles = [
            min(key for key in selected if key[0] == state)
            for state in sorted({key[0] for key in selected})
        ]
        indices = np.argsort(
            np.concatenate([selected[k]["order"] for k in first_tiles])
        )
        payload = {
            **payload,
            "states": payload["states"][indices],
            "points": payload["points"][indices],
            "farm": {v: d[indices] for v, d in payload["farm"].items()},
            "values": {v: d[indices] for v, d in payload["values"].items()},
        }
        output_results[self.result_name] = self.format_payload(algo, payload)

    def _combine_chunks(
        self, chunks: dict[tuple[int, int], dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge mean sums or state frames without recreating eliminated axes.

        Parameters
        ----------
        chunks
            Nonempty state/target-indexed payloads. Mean chunks already contain
            singleton-state weighted sums and input-order metadata.

        Returns
        -------
        payload
            Combined point fields, coordinates and overlay geometry. Mean fields
            keep one state row; returned labels retain original input order.

        Raises
        ------
        ValueError
            If target tiles disagree on states or mean chunks have different
            target coordinates.
        """
        if self.reduction != "mean":
            return super()._combine_chunks(chunks)
        slabs = []
        for state in sorted({key[0] for key in chunks}):
            tiles = [chunks[key] for key in sorted(chunks) if key[0] == state]
            if any(not np.array_equal(t["states"], tiles[0]["states"]) for t in tiles):
                raise ValueError(f"{self.name}: Target tiles have inconsistent states")
            slabs.append(
                {
                    **tiles[0],
                    "points": np.concatenate([t["points"] for t in tiles], axis=1),
                    "values": {
                        v: np.concatenate([t["values"][v] for t in tiles], axis=1)
                        for v in self.fields
                    },
                }
            )
        points = slabs[0]["points"]
        if any(
            s["points"].shape != points.shape
            or not np.allclose(s["points"], points, rtol=0, atol=1e-10)
            for s in slabs[1:]
        ):
            raise ValueError(f"{self.name}: Slice targets must be static across states")
        first = min(slabs, key=lambda s: s["first_rank"])
        states = np.array(["mean"])
        if self.ret_states:
            order = np.concatenate([s["order"] for s in slabs])
            states = np.concatenate([s["states"] for s in slabs])[np.argsort(order)]
        return {
            "states": states,
            "farm": first["farm"],
            "points": points,
            "values": {
                v: np.sum([s["values"][v] for s in slabs], axis=0) for v in self.fields
            },
        }

    def _grid_data(self, payload: dict[str, Any]) -> tuple[Any, Any, Any, np.ndarray]:
        """Recover rectangular plane coordinates from actual supplied targets.

        Parameters
        ----------
        payload
            Selected points with shape ``(state, point, 3)``.

        Returns
        -------
        x
            Projected x coordinates, scalar when normal to the selected plane.
        y
            Projected y coordinates, scalar when normal to the selected plane.
        z
            Projected z coordinates, scalar when normal to the selected plane.
        points
            First state row in original world-coordinate target order.

        Raises
        ------
        ValueError
            If targets move across states or do not form one rectangular plane.
        """
        points = payload["points"]
        if not np.allclose(points, points[:1], rtol=0, atol=1e-10):
            raise ValueError(f"{self.name}: Slice targets must be static across states")
        points = points[0]
        projected = grids.project_points(
            points, self.plane, self._grid_options.get("x_direction", 270)
        )
        axes = [np.unique(np.round(projected[:, i], 10)) for i in range(3)]
        active = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[self.plane]
        a, b, c = active
        if (
            len(axes[c]) != 1
            or len(axes[a]) * len(axes[b]) != len(points)
            or len(np.unique(np.round(projected[:, (a, b)], 10), axis=0)) != len(points)
        ):
            raise ValueError(
                f"{self.name}: Targets do not form a rectangular {self.plane} plane"
            )
        return tuple(axes[i] if i != c else axes[i][0] for i in range(3)) + (points,)

    def format_payload(self, algo: Algorithm, payload: dict[str, Any]) -> Any:
        """Format run-local values using constructor-defined reduction and names.

        Parameters
        ----------
        algo
            Executing algorithm, available to specialized renderers.
        payload
            Combined selected states, static target coordinates and point fields.

        Returns
        -------
        data
            Data in the selected NumPy, pandas or xarray representation, reduced
            over states when configured. If requested, append state labels and
            then static grid metadata in a tuple. Write ``to_file`` when set.
        """
        result = self._format_data(payload)
        if self.to_file is not None:
            self._write_data(result)
        returned = [result]
        if self.ret_states:
            returned.append(payload["states"])
        if self.ret_grid:
            returned.append(self._grid_data(payload))
        return returned[0] if len(returned) == 1 else tuple(returned)

    def _format_data(self, payload: dict[str, Any]) -> Any:
        """Format fields without writing intermediate plane datasets.

        Parameters
        ----------
        payload
            Combined point fields, static coordinates and selected state labels.

        Returns
        -------
        data
            NumPy, pandas or xarray representation with normalized plane
            coordinates. State means omit the state axis; state-resolved data
            retains it. Source coordinates and field arrays are not mutated.
        """
        # Select the actual point order and normalize independent coordinate copies.
        grid = self._grid_data(payload)
        positions = dict(zip("xyz", grid[:3]))
        a_name, b_name = self.plane[0], self.plane[1]
        c_name = next(c for c in "xyz" if c not in self.plane)
        a = positions[a_name] / self.normalize.get(a_name, 1)
        b = positions[b_name] / self.normalize.get(b_name, 1)
        c = positions[c_name] / self.normalize.get(c_name, 1)
        data = self._field_data(payload)
        if self.reduction == "mean":
            data = {v: d[0] for v, d in data.items()}

        # Convert at finalization so workers never construct output datasets.
        suffix = "p" if self.reduction == "mean" else "sp"
        args = (data, a, b) if suffix == "p" else (data, payload["states"], a, b)
        if self.data_format == "numpy":
            return getattr(grids, f"np2np_{suffix}")(*args)
        if self.data_format == "pandas":
            return getattr(grids, f"np2pd_{suffix}")(*args, self.plane, self.label_map)
        return getattr(grids, f"np2xr_{suffix}")(*args, c, self.plane, self.label_map)

    def _field_data(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        """Order supplied targets and apply configured field transforms.

        Parameters
        ----------
        payload
            Combined point fields and coordinates in supplied target order.

        Returns
        -------
        values
            Selected ``(state, point)`` arrays in rectangular plane order, with
            configured field transforms applied to independent arrays.
        """
        projected = grids.project_points(
            payload["points"][0],
            self.plane,
            self._grid_options.get("x_direction", 270),
        )
        a, b = ("xyz".index(axis) for axis in self.plane)
        order = np.lexsort(
            (np.round(projected[:, b], 10), np.round(projected[:, a], 10))
        )
        data = {v: payload["values"][v][:, order] for v in self.fields}
        return {
            v: np.clip(
                d / self.normalize_v.get(v, 1),
                self.vmin.get(v, -np.inf),
                self.vmax.get(v, np.inf),
            )
            for v, d in data.items()
        }

    def _required_variables(self) -> tuple[str, ...]:
        """Request state weights only for weighted mean reductions."""
        variables = super()._required_variables()
        if self.reduction == "mean":
            variables = tuple(dict.fromkeys([*variables, FV.WEIGHT]))
        return variables

    def _write_data(self, data: Any) -> None:
        """Write the selected representation without retaining the result.

        Parameters
        ----------
        data
            Final NumPy, pandas or xarray result. The configured format selects
            its writer, destination and ``write_pars`` options.
        """
        if self.to_file is None:
            raise ValueError(f"{self.name}: No output filename configured")
        path = self.get_fpath(self.to_file)
        if self.data_format == "xarray":
            write_nc(data, path, **self.write_pars)
        elif self.data_format == "pandas":
            self.write(str(self.to_file), data, **self.write_pars)
        else:
            data.tofile(path, **self.write_pars)
