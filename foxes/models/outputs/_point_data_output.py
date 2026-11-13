"""Shared numerical point-grid collection without rendering or animation."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import Index

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, Output, TData
from foxes.core.engine import get_engine
from foxes.utils import Dict

if TYPE_CHECKING:
    from foxes.core import Algorithm, Model, PointDataModelList
    from foxes.core.model import LoadedData


class PointDataOutput(Output):
    """Collect configured point grids in run-local, engine-owned chunk payloads.

    Retain point/grid configuration without preparing scientific data.
    """

    def __init__(
        self,
        plane: str = "xy",
        variables: list[str] | None = None,
        chunk: int | tuple[int, int] | list[int | tuple[int, int]] | None = None,
        states_sel: list[Any] | None = None,
        states_isel: list[int] | None = None,
        grid_pars: dict[str, Any] | None = None,
        ambient: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        plane
            Grid orientation: ``"xy"``, ``"xz"`` or ``"yz"``.
        variables
            Requested point fields; ``None`` selects wind speed.
        chunk
            State-chunk index, state/target chunk pair, or a list of either.
            ``None`` selects every chunk.
        states_sel
            State labels to retain, mutually exclusive with ``states_isel``.
        states_isel
            Nonnegative positions in the input states.
        grid_pars
            Plane-specific grid bounds, spacing and normal-axis position.
        ambient
            Skip wake calculations on generated grids.
        kwargs
            Common output paths and model options.
        """
        super().__init__(
            mdata_variables=None, fdata_variables=None, tdata_variables=None, **kwargs
        )
        self.plane = plane
        self.variables = variables
        self.chunk = chunk
        self.states_sel = states_sel
        self.states_isel = states_isel
        self.grid_pars = grid_pars
        self.ambient = ambient
        self._point_models: PointDataModelList | None = None

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Validate configuration and prepare fields, selections and submodels.

        Parameters
        ----------
        algo
            Farm algorithm whose wake models calculate requested point fields.
        loaded_data
            Shared initialization data to extend.
        force
            Reinitialize the output and its point models.
        verbosity
            Initialization verbosity.

        Returns
        -------
        loaded_data
            Combined loaded model data.

        Raises
        ------
        ValueError
            If the plane, variables or selections are invalid,
            or initialization is attempted while running.
        """
        # Reject invalid configuration before loading scientific model data.
        if self.running:
            raise ValueError(f"{self.name}: Cannot initialize while running")
        if not self.initialized or force:
            self.chunks = tuple(self._validated_selections())
            self._variables = tuple(
                [FV.WS] if self.variables is None else self.variables
            )
            if not self._variables or any(
                not isinstance(v, str) or not v for v in self._variables
            ):
                raise ValueError(
                    f"{self.name}: variables must contain nonempty field names"
                )
            self._grid_options = deepcopy(self.grid_pars or {})
            self._result_key = self.var("slice_chunks")
            self._grid_key = self.var("slice_points")
            self._selection_key = self.var("slice_selection")
            self._chunk_size_points: int | None = None

            # Build scientific dependencies only during initialization.
            self._point_models, self._point_parameters = algo._collect_point_models(
                ambient=self.ambient
            )
        return super().initialize(algo, loaded_data, force, verbosity)

    def _validated_selections(self) -> list[int | tuple[int, int] | None]:
        """Validate plane, chunk selections and mutually exclusive state selectors.

        Returns
        -------
        selections
            Configured state-chunk indices or state/target pairs. A ``None``
            entry selects every chunk; the configuration itself is not mutated.

        Raises
        ------
        ValueError
            If the plane is unknown, both state selectors are supplied, or
            chunk/state positions are not nonnegative integers.
        """
        if self.plane not in ("xy", "xz", "yz"):
            raise ValueError(f"{self.name}: Unknown plane '{self.plane}'")
        if self.states_sel is not None and self.states_isel is not None:
            raise ValueError(f"{self.name}: Choose states_sel or states_isel")
        selections: list[int | tuple[int, int] | None] = (
            list(self.chunk) if isinstance(self.chunk, list) else [self.chunk]
        )
        for selection in selections:
            indices = selection if isinstance(selection, tuple) else (selection,)
            if selection is not None and (
                len(indices) not in (1, 2)
                or isinstance(selection, tuple)
                and len(indices) != 2
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or i < 0
                    for i in indices
                )
            ):
                raise ValueError(f"{self.name}: Invalid chunk selection {selection!r}")
        if self.states_isel is not None and any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0
            for i in self.states_isel
        ):
            raise ValueError(
                f"{self.name}: State positions must be nonnegative integers"
            )
        return selections

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Capture the point limit and stash constructor data for execution.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Shared lifecycle stash for scientific submodels.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        # Refresh the limit for reused outputs before workers see the model.
        if self.running:
            raise ValueError(f"{self.name}: Cannot set_running while running")
        engine = get_engine(error=False)
        self._chunk_size_points = None if engine is None else engine.chunk_size_points
        if self._chunk_size_points is not None and self._chunk_size_points <= 0:
            raise ValueError(f"{self.name}: chunk_size_points must be positive")
        super().set_running(algo, data_stash, sel, verbosity)

        # Keep raw selections and unused input dictionaries out of worker models.
        configuration = {
            key: vars(self).pop(key)
            for key in (
                "chunk",
                "variables",
                "states_sel",
                "states_isel",
                "grid_pars",
            )
        }
        if data_stash is not None:
            data_stash[self.name]["configuration"] = configuration

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore constructor configuration after output evaluation.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Parent lifecycle stash, or ``None`` for discarded configuration.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        # Restore caller-owned configuration only after leaving running state.
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            vars(self).update(data_stash[self.name].pop("configuration"))

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Load shared grid coordinates and a partial state-selection variable.

        Parameters
        ----------
        algo
            Algorithm supplying all turbine positions and hub heights.
        loaded_data
            Initialization mapping extended with static points in ``extra_data``
            and, when needed, a boolean ``(FC.STATE,)`` selection in ``data_vars``.
        force
            Overwrite previously loaded model data.
        verbosity
            Initialization verbosity.
        """
        from . import grids
        from ._point_calculation import farm_bounds

        super().load_data(algo, loaded_data, force, verbosity)
        bounds = farm_bounds(algo, self.name)
        grid_pars = deepcopy(self.grid_pars or {})
        if grid_pars.get("n_img_points") is None:
            grid_pars.setdefault("resolution", 100.0)
        points = getattr(grids, f"get_grid_{self.plane}")(bounds, **grid_pars)[-1]
        points.setflags(write=False)
        loaded_data["extra_data"][self._grid_key] = points

        # Partial selections follow the state dimension and are sliced by engines.
        states = Index(algo.states.index())
        selection = None
        if self.states_sel is not None:
            positions = states.get_indexer(self.states_sel)
            if np.any(positions < 0):
                raise ValueError(f"{self.name}: Selected state labels are unavailable")
            selection = np.zeros(len(states), dtype=bool)
            selection[positions] = True
        elif self.states_isel is not None:
            if any(i >= len(states) for i in self.states_isel):
                raise ValueError(f"{self.name}: State position outside input states")
            selection = np.zeros(len(states), dtype=bool)
            selection[self.states_isel] = True
        self._selection = None
        loaded_data["data_vars"].pop(self._selection_key, None)
        if selection is not None and not np.all(selection):
            self._selection = self._selection_key
            loaded_data["data_vars"][self._selection_key] = ((FC.STATE,), selection)

    def get_calc_points(self, algo: Algorithm) -> np.ndarray | None:
        """Return the static flow grid used by ``calc_outputs``.

        Parameters
        ----------
        algo
            Initialized algorithm owning the grid in loaded extra data.

        Returns
        -------
        points
            Read-only grid coordinates with shape ``(n_points, 3)``.

        Raises
        ------
        ValueError
            If the output has not been initialized or is currently running.
        """
        if not self.initialized or self.running:
            raise ValueError(
                f"{self.name}: Calculation points require an initialized, "
                "non-running output"
            )
        return algo.loaded_data["extra_data"][self._grid_key]

    def sub_models(self) -> list[Model]:
        """Return point models participating in the normal model lifecycle.

        Returns
        -------
        models
            The initialized point-model chain, if present.
        """
        # Propagate initialization and running state to scientific dependencies.
        return [] if self._point_models is None else [self._point_models]

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect supplied targets or evaluate shared-grid blocks for one chunk.

        Parameters
        ----------
        algo
            Running farm algorithm.
        mdata
            Complete chunk model data in prepared farm order.
        fdata
            Complete chunk farm data in farm order.
        tdata
            Existing target coordinates and fields, or ``None`` for a farm run.
        model_results
            Scientific calculation returns, unused here.
        chunk_results
            Shared output payload mapping to extend.

        Returns
        -------
        chunk_results
            Mapping containing private NumPy flow-grid payloads for selected chunks.

        Raises
        ------
        KeyError
            If a required target field or shared grid is unavailable.
        """
        # Leave other outputs untouched and skip unselected chunk tasks.
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        key = (mdata.chunki_states, mdata.chunki_points)
        if not any(c is None or c == key or c == key[0] for c in self.chunks):
            return results
        selection = None if self._selection is None else mdata[self._selection]
        if selection is not None and not np.any(selection):
            results[self._result_key] = None
            return results

        # Capture farm-order metadata without retaining writable source arrays.
        farm = {
            v: np.broadcast_to(d, (fdata.n_states, fdata.n_turbines)).copy()
            for v, d in fdata.items()
            if v in self._farm_variables() and fdata.dims[v] == (FC.STATE, FC.TURBINE)
        }
        if tdata is None:
            points, values = self._calculate_grid(algo, mdata, fdata)
        else:
            points, values = self._collect_targets(mdata, tdata)
        payload: dict[str, Any] = dict(
            states=mdata[FC.STATE].copy(),
            farm=farm,
            points=points,
            values=values,
        )
        if selection is not None:
            payload["states"] = payload["states"][selection]
            if payload["points"].shape[0] != 1:
                payload["points"] = payload["points"][selection]
            for mapping in (payload["farm"], payload["values"]):
                for variable in mapping:
                    mapping[variable] = mapping[variable][selection]
        results[self._result_key] = payload
        return results

    def _collect_targets(
        self,
        mdata: MData,
        tdata: TData,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Copy supplied coordinates and fields in target/target-point order.

        Parameters
        ----------
        mdata
            Metadata defining the current state count.
        tdata
            Supplied target data, read without mutation.

        Returns
        -------
        points
            Flattened ``(state, target * tpoint, 3)`` coordinates, with a singleton
            state axis when positions do not change between states.
        values
            Required fields as copied ``(state, target * tpoint)`` arrays.

        Raises
        ------
        KeyError
            If a required scientific point field is absent.
        """
        # Expand broadcast axes before flattening target/target-point dimensions.
        shape = (mdata.n_states, tdata.n_targets, tdata.n_tpoints)
        points = np.broadcast_to(tdata[FC.TARGETS], (*shape, 3))
        if np.array_equal(points, np.broadcast_to(points[:1], points.shape)):
            points = points[:1]
        values = {}
        for variable in self._required_variables():
            if variable not in tdata:
                raise KeyError(f"{self.name}: Missing target field '{variable}'")
            values[variable] = (
                np.broadcast_to(tdata[variable], shape).reshape(shape[0], -1).copy()
            )
        return points.reshape(points.shape[0], -1, 3).copy(), values

    def _calculate_grid(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Evaluate loaded grid points in bounded private target containers.

        Parameters
        ----------
        algo
            Running algorithm supplying scientific point-model context.
        mdata
            Chunk metadata and shared grid coordinates in ``extra_data``.
        fdata
            Farm-ordered scientific inputs, borrowed without mutation.

        Returns
        -------
        points
            Shared static coordinates with shape ``(1, point, 3)``.
        values
            Required field arrays with shape ``(state, point)``.

        Raises
        ------
        RuntimeError
            If scientific point dependencies were not initialized.
        """
        from ._point_calculation import point_blocks

        # Keep shared extra-data arrays out of expensive deep copies.
        if self._point_models is None:
            raise RuntimeError(f"{self.name}: Initialize output before calculating")
        points = mdata.extra_data[self._grid_key]
        variables = self._required_variables()
        blocks: dict[str, list[np.ndarray]] = {v: [] for v in variables}

        # Release each temporary target container before evaluating the next block.
        for target_data in point_blocks(
            algo,
            mdata,
            fdata,
            points,
            self._point_models,
            self._point_parameters,
            self._chunk_size_points,
        ):
            for variable in variables:
                blocks[variable].append(target_data[variable][:, :, 0].copy())
        return (
            points[None, :, :],
            {v: np.concatenate(parts, axis=1) for v, parts in blocks.items()},
        )

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Buffer private chunk payloads without rendering or writing.

        Parameters
        ----------
        algo
            Calculation algorithm; unused by buffering.
        n_chunks_states
            Number of state chunks.
        n_chunks_targets
            Number of target chunks.
        new_chunk_results
            Pending chunk payloads, including other outputs.
        output_results
            Run-local buffers, keyed by this output's private result name.
        """
        # Keep all run-local aggregation outside model instances.
        output_results.setdefault(self._result_key, {}).update(
            new_chunk_results.pop(self._result_key, {})
        )

    def _selected_chunks(
        self,
        chunks: dict[tuple[int, int], Any],
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Validate selected chunk indices and discard empty state selections.

        Parameters
        ----------
        chunks
            Payloads keyed by state/target chunk index. ``None`` payloads denote
            chunks whose states were excluded by the configured selection.
        n_chunks_states
            Total number of engine state chunks.
        n_chunks_targets
            Total number of engine target chunks.

        Returns
        -------
        selected
            Nonempty payloads with unchanged chunk keys and borrowed arrays.

        Raises
        ------
        ValueError
            If an explicit chunk selection is absent or expected tiles are missing.
        """
        # Validate explicit chunk indices rather than silently ignoring mistakes.
        for selection in self.chunks:
            if selection is not None and not any(
                selection == key or selection == key[0] for key in chunks
            ):
                raise ValueError(
                    f"{self.name}: Selected chunk {selection} is unavailable"
                )
        expected = {
            (s, t)
            for s in range(n_chunks_states)
            for t in range(n_chunks_targets)
            if any(c is None or c == (s, t) or c == s for c in self.chunks)
        }
        if set(chunks) != expected:
            raise ValueError(
                f"{self.name}: Missing slice chunks {expected - set(chunks)}"
            )
        return {key: payload for key, payload in chunks.items() if payload is not None}

    def _combine_chunks(
        self,
        chunks: dict[tuple[int, int], dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble selected chunks in engine order, not result-arrival order.

        Parameters
        ----------
        chunks
            Nonempty state/target-indexed payloads with state coordinates,
            farm overlays, flattened target points and scientific fields.

        Returns
        -------
        payload
            State-ordered labels, overlays and ``(state, point)`` fields, with
            full ``(state, point, 3)`` coordinates. Source arrays are not mutated.

        Raises
        ------
        ValueError
            If target tiles disagree on states or state slabs have different
            point counts.
        """
        from ._point_calculation import join_point_tiles

        slabs = []
        for state in sorted({key[0] for key in chunks}):
            tiles = [chunks[key] for key in sorted(chunks) if key[0] == state]
            slabs.append(join_point_tiles(tiles, self.name))
        if len({s["points"].shape[1] for s in slabs}) != 1:
            raise ValueError(
                f"{self.name}: Selected chunks have different point counts"
            )

        # Concatenate identically shaped state slabs without sorting state labels.
        return dict(
            states=np.concatenate([s["states"] for s in slabs]),
            farm={
                v: np.concatenate([s["farm"][v] for s in slabs])
                for v in slabs[0]["farm"]
            },
            points=np.concatenate([s["points"] for s in slabs]),
            values={
                v: np.concatenate([s["values"][v] for s in slabs])
                for v in slabs[0]["values"]
            },
        )

    def _required_variables(self) -> tuple[str, ...]:
        """Return only point fields required by this numerical output."""
        return tuple(dict.fromkeys(self._variables))

    def _farm_variables(self) -> tuple[str, ...]:
        """Return farm metadata needed by final rendering, if any."""
        return ()
