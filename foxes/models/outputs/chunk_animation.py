"""Collect flow chunks and save unified XY animations during finalization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from pandas import Index
from xarray import Dataset

from foxes.core import Output, TData, MData, FData
from foxes.core.engine import get_engine
from foxes.utils import Dict
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from foxes.core import Algorithm, Model, PointDataModelList
    from foxes.core.model import LoadedData


class ChunkAnimationOutput(Output):
    """Write one state-ordered XY flow animation per variable.

    Supplied target data is reused without further scientific calculations.
    Farm-only runs evaluate a shared grid in bounded point blocks. Workers
    return independent payloads; only finalization creates figures and files.
    """

    def __init__(
        self,
        chunk: int | tuple[int, int] | list[int | tuple[int, int]] | None = None,
        variables: list[str] | None = None,
        base_name: str = "chunk_animation.gif",
        fps: int = 4,
        figsize: tuple[float, float] = (8, 8),
        states_sel: list[Any] | None = None,
        states_isel: list[int] | None = None,
        grid_pars: dict[str, Any] | None = None,
        plot_pars: dict[str, Any] | None = None,
        ambient: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure optional chunk animations.

        Parameters
        ----------
        chunk
            State-chunk index, exact state/target chunk pair, or a list of either.
            ``None`` selects every chunk. Selected target tiles are joined before
            state chunks; explicit indices must exist in the calculation.
        variables
            Point fields to animate; ``None`` selects wind speed.
        base_name
            Filename including ``.gif`` or ``.mp4``. Files append the variable
            before the suffix, without a chunk suffix. The final suffix, after
            ``out_fname_fun``, selects Pillow for GIF or ffmpeg for MP4.
        fps
            Positive frame rate. GIF uses Pillow; MP4 requires external ffmpeg.
        figsize
            Figure width and height in inches.
        states_sel
            Input state labels to retain, preserving calculation order.
        states_isel
            Nonnegative positions in the input states, preserving calculation order.
            Mutually exclusive with ``states_sel``.
        grid_pars
            Options for ``get_grid_xy``, such as ``resolution``, ``n_img_points``,
            ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``z``, ``xspace`` and ``yspace``.
        plot_pars
            Options for ``FlowPlots2D.gen_states_fig_xy``, including ``vmin``,
            ``vmax``, ``cmap``, ``levels`` and ``quiver_n`` for rectangular XY
            grids. Other targets use Matplotlib ``scatter`` options, with
            ``vmin`` and ``vmax`` selecting the shared colour scale.
        ambient
            Skip wake calculations on generated grids. Supplied target data is
            always used as given, independently of this option.
        kwargs
            Common output options, including ``out_dir`` and ``out_fname_fun``.
        """
        # Retain configuration only; initialization validates and prepares it.
        super().__init__(
            mdata_variables=None, fdata_variables=None, tdata_variables=None, **kwargs
        )
        self.chunk = chunk
        self.variables = variables
        self.base_name = base_name
        self.fps = fps
        self.figsize = figsize
        self.states_sel = states_sel
        self.states_isel = states_isel
        self.grid_pars = grid_pars
        self.plot_pars = plot_pars
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
            Farm algorithm whose wake models calculate animation fields.
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
            If file format, frame rate, variables or selections are invalid,
            or initialization is attempted while running.
        """
        # Reject invalid configuration before loading scientific model data.
        if self.running:
            raise ValueError(f"{self.name}: Cannot initialize while running")
        if not self.initialized or force:
            if Path(self.base_name).suffix.lower() not in (".gif", ".mp4"):
                raise ValueError(f"{self.name}: base_name must end in .gif or .mp4")
            if (
                isinstance(self.fps, bool)
                or not isinstance(self.fps, int)
                or self.fps <= 0
            ):
                raise ValueError(f"{self.name}: fps must be a positive integer")
            if self.states_sel is not None and self.states_isel is not None:
                raise ValueError(f"{self.name}: Choose states_sel or states_isel")
            selections = self.chunk if isinstance(self.chunk, list) else [self.chunk]
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
                    raise ValueError(
                        f"{self.name}: Invalid chunk selection {selection!r}"
                    )
            if self.states_isel is not None and any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in self.states_isel
            ):
                raise ValueError(
                    f"{self.name}: State positions must be nonnegative integers"
                )

            # Snapshot mutable settings before this output enters running state.
            self.chunks = tuple(selections)
            self._variables = tuple(
                [FV.WS] if self.variables is None else self.variables
            )
            if not self._variables or any(
                not isinstance(v, str) or not v for v in self._variables
            ):
                raise ValueError(
                    f"{self.name}: variables must contain nonempty field names"
                )
            self._base_name = Path(self.base_name)
            self._plot_pars = deepcopy(self.plot_pars or {})
            self._result_key = self.var("chunk_animation")
            self._grid_key = self.var("animation_points")
            self._selection_key = self.var("animation_selection")
            self._chunk_size_points: int | None = None

            # Build scientific dependencies only during initialization.
            self._point_models, self._point_parameters = algo._collect_point_models(
                ambient=self.ambient
            )
        return super().initialize(algo, loaded_data, force, verbosity)

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
                "plot_pars",
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
        # Bound the grid using every position, including state-dependent layouts.
        from foxes.output.grids import get_grid_xy

        super().load_data(algo, loaded_data, force, verbosity)
        types = algo.farm_controller.turbine_types
        if types is None or not algo.farm.turbines:
            raise ValueError(
                f"{self.name}: Grid generation requires initialized turbines"
            )
        xy = np.concatenate([t.xy.reshape(-1, 2) for t in algo.farm.turbines])
        heights = [
            t.H if t.H is not None else types[i].H
            for i, t in enumerate(algo.farm.turbines)
        ]
        if any(h is None for h in heights):
            raise ValueError(
                f"{self.name}: Grid generation requires turbine hub heights"
            )
        bounds = Dataset(
            {
                FV.X: ("position", xy[:, 0]),
                FV.Y: ("position", xy[:, 1]),
                FV.H: ("turbine", np.asarray(heights, dtype=float)),
            }
        )
        grid_pars = deepcopy(self.grid_pars or {})
        if grid_pars.get("n_img_points") is None:
            grid_pars.setdefault("resolution", 100.0)
        loaded_data["extra_data"][self._grid_key] = get_grid_xy(bounds, **grid_pars)[-1]

        # Partial selections follow the state dimension and are sliced by engines.
        states = Index(algo.states.index())
        selection = None
        if self.states_sel is not None:
            if not np.all(Index(self.states_sel).isin(states)):
                raise ValueError(f"{self.name}: Selected state labels are unavailable")
            selection = states.isin(self.states_sel)
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
            if v in (FV.X, FV.Y, FV.H, FV.D, FV.YAW, FV.AMB_WD, FV.YAWM)
            and fdata.dims[v] == (FC.STATE, FC.TURBINE)
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
        """Copy actual target points, retaining all target quadrature locations."""
        # Expand broadcast axes before flattening target/target-point dimensions.
        shape = (mdata.n_states, tdata.n_targets, tdata.n_tpoints)
        points = np.broadcast_to(tdata[FC.TARGETS], (*shape, 3))
        values = {}
        for variable in dict.fromkeys([*self._variables, FV.WD, FV.WS]):
            if variable not in tdata:
                raise KeyError(f"{self.name}: Missing target field '{variable}'")
            values[variable] = (
                np.broadcast_to(tdata[variable], shape).reshape(shape[0], -1).copy()
            )
        return points.reshape(shape[0], -1, 3).copy(), values

    def _calculate_grid(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Evaluate shared points in bounded blocks on private chunk containers."""
        # Keep shared extra-data arrays out of expensive deep copies.
        if self._point_models is None:
            raise RuntimeError(f"{self.name}: Initialize output before calculating")
        md = deepcopy(mdata, {id(mdata.extra_data): mdata.extra_data})
        fd = deepcopy(fdata)
        points = mdata.extra_data[self._grid_key]
        size = self._chunk_size_points or len(points)
        variables = list(dict.fromkeys([*self._variables, FV.WD, FV.WS]))
        blocks: dict[str, list[np.ndarray]] = {v: [] for v in variables}

        # Restore downwind order on private containers for point wake evaluation.
        order = fd[FV.ORDER]
        for data in (md, fd):
            for variable, dims in data.dims.items():
                if dims[:2] == (FC.STATE, FC.TURBINE) and variable != FV.ORDER:
                    values = data[variable]
                    if values.shape[1] > 1:
                        data[variable] = values[np.arange(len(order))[:, None], order]

        # Release each temporary target container before evaluating the next block.
        for start in range(0, len(points), size):
            block = points[start : start + size]
            target_data = TData.from_points(
                np.broadcast_to(block, (md.n_states, *block.shape)),
                mdata=md,
            )
            calculated = self._point_models.calculate(
                algo,
                md,
                fd,
                target_data,
                parameters=self._point_parameters,
            )
            for variable in variables:
                blocks[variable].append(calculated[variable][:, :, 0].copy())
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

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Combine target tiles and state chunks, then save one file per variable.

        Parameters
        ----------
        algo
            Algorithm supplying renderer metadata and verbosity.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total target chunks.
        output_results
            Run-local buffers; the private buffer is removed after successful
            writing. Other outputs' results remain untouched.

        Raises
        ------
        ValueError
            If selected chunks are absent or target-tile states disagree.
        """
        # Validate explicit chunk indices rather than silently ignoring mistakes.
        chunks = output_results.get(self._result_key, {})
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
                f"{self.name}: Missing animation chunks {expected - set(chunks)}"
            )
        chunks = {
            key: payload for key, payload in chunks.items() if payload is not None
        }
        if not chunks:
            output_results.pop(self._result_key, None)
            return

        # Assemble already selected numerical frames before rendering once.
        payload = self._combine_chunks(chunks)
        self._render(algo, payload)
        output_results.pop(self._result_key, None)

    def _combine_chunks(
        self,
        chunks: dict[tuple[int, int], dict[str, Any]],
    ) -> dict[str, Any]:
        """Join target tiles before state slabs, independently of arrival order."""
        # Keep target point order and reject inconsistent state coordinates.
        slabs = []
        for state in sorted({key[0] for key in chunks}):
            tiles = [chunks[key] for key in sorted(chunks) if key[0] == state]
            if any(not np.array_equal(t["states"], tiles[0]["states"]) for t in tiles):
                raise ValueError(f"{self.name}: Target tiles have inconsistent states")
            slabs.append(
                dict(
                    states=tiles[0]["states"],
                    farm=tiles[0]["farm"],
                    points=np.concatenate(
                        [
                            np.broadcast_to(
                                t["points"], (len(t["states"]), *t["points"].shape[1:])
                            )
                            for t in tiles
                        ],
                        axis=1,
                    ),
                    values={
                        v: np.concatenate([t["values"][v] for t in tiles], axis=1)
                        for v in tiles[0]["values"]
                    },
                )
            )
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

    def _render(self, algo: Algorithm, payload: dict[str, Any]) -> None:
        """Render combined fields once and always close figures on failures."""
        # Import plotting only for selected payloads, never in chunk workers.
        import matplotlib.pyplot as plt
        from foxes.output import Animator, FlowPlots2D
        from foxes.output.grids import np2np_sp
        from foxes.output.rose_plot import RosePlotOutput

        farm = Dataset(
            {v: ((FC.STATE, FC.TURBINE), d) for v, d in payload["farm"].items()},
            coords={FC.STATE: payload["states"]},
        )
        points = payload["points"]
        x, y = np.unique(points[0, :, 0]), np.unique(points[0, :, 1])
        grid = (x, y, points[0, 0, 2], points[0])
        values = payload["values"]
        rectangular = (
            len(x) * len(y) == points.shape[1]
            and len(x) > 1
            and len(y) > 1
            and np.all(points == points[0])
            and np.all(points[0, :, 2] == points[0, 0, 2])
            and len(np.unique(points[0, :, :2], axis=0)) == points.shape[1]
        )
        if rectangular:
            order = np.lexsort((points[0, :, 1], points[0, :, 0]))
            gridded = np2np_sp(
                {v: d[:, order] for v, d in values.items()},
                payload["states"],
                x,
                y,
            )

        # Save one file per variable with a figure-local cleanup boundary.
        plotter = FlowPlots2D(algo, farm)
        for variable in self._variables:
            options = dict(self._plot_pars)
            options.setdefault("vlabel", RosePlotOutput.get_data_info(variable)[1])
            parameters = dict(
                var=variable,
                variables=list(values),
                data_format="numpy",
                vmin=options.pop("vmin", None),
                vmax=options.pop("vmax", None),
            )
            filename = self._base_name.with_name(
                f"{self._base_name.stem}_{variable}{self._base_name.suffix}"
            )
            path = self.get_fpath(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            if algo.verbosity > 0:
                print(f"{self.name}: Writing {path}")

            # Ensure renderer failures cannot leak Matplotlib figures.
            fig, ax = plt.subplots(figsize=self.figsize)
            try:
                animator = Animator(fig=fig)
                if rectangular:
                    generator = plotter.gen_states_fig_xy(
                        (parameters, gridded, payload["states"], grid),
                        fig=fig,
                        ax=ax,
                        animated=True,
                        ret_im=True,
                        **options,
                    )
                else:
                    generator = self._scatter_frames(
                        fig,
                        ax,
                        payload,
                        variable,
                        parameters,
                        options,
                    )
                animator.add_generator(generator)
                animation = animator.animate(verbosity=0)
                animation.save(
                    filename=path,
                    writer="pillow" if path.suffix.lower() == ".gif" else "ffmpeg",
                    fps=self.fps,
                )
            finally:
                plt.close(fig)

    def _scatter_frames(
        self,
        fig: Figure,
        ax: Axes,
        payload: dict[str, Any],
        variable: str,
        parameters: dict[str, Any],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Show arbitrary or moving target locations without resampling fields."""
        # Use fixed axes and a shared color scale so frames remain comparable.
        from matplotlib.colors import Normalize

        points, values = payload["points"], payload["values"][variable]
        norm = Normalize(
            np.min(values) if parameters["vmin"] is None else parameters["vmin"],
            np.max(values) if parameters["vmax"] is None else parameters["vmax"],
        )
        label = options.pop("vlabel")
        ax.set(xlabel="x [m]", ylabel="y [m]")
        ax.update_datalim(points[:, :, :2].reshape(-1, 2))
        ax.autoscale_view()
        for index, state in enumerate(payload["states"]):
            artist = ax.scatter(
                points[index, :, 0],
                points[index, :, 1],
                c=values[index],
                norm=norm,
                **options,
            )
            if index == 0:
                fig.colorbar(artist, ax=ax, label=label)
            title = ax.text(
                0.5,
                1.01,
                f"State {state}",
                transform=ax.transAxes,
                ha="center",
            )
            yield fig, [artist, title]
