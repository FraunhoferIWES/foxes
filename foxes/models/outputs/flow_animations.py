"""Collect flow chunks and save unified flow and turbine animations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

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


class Animator:
    """Compose an animation from generators yielding figures and artists."""

    def __init__(self, fig: Figure | None = None) -> None:
        """Configure the shared figure.

        Parameters
        ----------
        fig
            Shared figure, or ``None`` to use the first yielded figure.
        """
        self.fig = fig
        self._gens: list[Any] = []

    def add_generator(self, gen: Any) -> None:
        """Add a synchronized artist generator.

        Parameters
        ----------
        gen
            Generator yielding ``(figure, artists)`` pairs.
        """
        self._gens.append(gen)

    @property
    def generators(self) -> list[Any]:
        """The synchronized artist generators.

        Returns
        -------
        generators
            Generators yielding figures and artist collections.
        """
        return self._gens

    def animate(self, verbosity: int = 1, **kwargs: Any) -> Any:
        """Create a Matplotlib artist animation.

        Parameters
        ----------
        verbosity
            Diagnostic verbosity; zero is silent.
        kwargs
            Options for ``matplotlib.animation.ArtistAnimation``.

        Returns
        -------
        animation
            Composed animation, or ``None`` when no generators were added.

        Raises
        ------
        ValueError
            If a generator yields an invalid pair or a different figure.
        """
        from matplotlib.animation import ArtistAnimation

        if len(self.generators) == 0:
            return None

        if verbosity > 0:
            print("Creating animation data")

        # Consume generators in lockstep until no artists remain.
        si = 0
        arts = []
        while True:
            if verbosity > 1:
                print(f"  Frame {si}")

            harts = []
            for g in self.generators:
                try:
                    y = next(g)

                    if len(y) != 2:
                        raise ValueError(
                            f"Expecting yield (fig, artists) from generator {g}"
                        )

                    fig, artists = y
                    if self.fig is None:
                        self.fig = fig
                    elif fig is not self.fig:
                        raise ValueError(f"Wrong figure returned by generator {g}")

                    harts += [a for a in artists]

                except StopIteration:
                    pass

            if len(harts):
                arts.append(harts)
                si += 1
            else:
                break

        if verbosity > 1:
            print("Done.")

        # Retain established playback defaults unless explicitly overridden.
        kwa = dict(interval=200, blit=True, repeat_delay=2000)
        kwa.update(kwargs)
        return ArtistAnimation(fig, arts, **kwa)


class FlowAnimation(Output):
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
        fps: float = 4,
        figsize: tuple[float, float] = (8, 8),
        states_sel: list[Any] | None = None,
        states_isel: list[int] | None = None,
        grid_pars: dict[str, Any] | None = None,
        plot_pars: dict[str, Any] | None = None,
        ambient: bool = False,
        layout_pars: dict[str, Any] | None = None,
        verbosity: int | None = None,
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
            Filename including ``.gif`` or ``.mp4``. Each file appends the
            animated variable before the suffix, even for one variable,
            without a chunk suffix. The final suffix, after
            ``out_fname_fun``, selects Pillow for GIF or ffmpeg for MP4.
        fps
            Positive finite frame rate; fractional rates are supported.
            GIF uses Pillow; MP4 requires external ffmpeg.
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
        layout_pars
            Options for ``FarmLayoutOutput.get_figure`` on the flow axes.
            ``None`` disables annotations; do not supply ``fig`` or ``ax``.
        verbosity
            Writing-message verbosity. ``None`` uses ``algo.verbosity``;
            positive values announce each destination, and zero suppresses it.
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
        self.layout_pars = layout_pars
        self.verbosity = verbosity
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
                or not isinstance(self.fps, (int, float))
                or not np.isfinite(self.fps)
                or self.fps <= 0
            ):
                raise ValueError(f"{self.name}: fps must be a positive finite number")
            if self.states_sel is not None and self.states_isel is not None:
                raise ValueError(f"{self.name}: Choose states_sel or states_isel")
            if self.layout_pars is not None and {"fig", "ax"} & self.layout_pars.keys():
                raise ValueError(f"{self.name}: layout_pars must not supply fig or ax")
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
            self._layout_pars = deepcopy(self.layout_pars)
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
                "layout_pars",
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

    def _render(
        self,
        algo: Algorithm,
        payload: dict[str, Any],
        extra_generators: tuple[Callable[[Axes], Iterator[Any]], ...] = (),
    ) -> None:
        """Render combined fields once and always close figures on failures."""
        # Import plotting only for selected payloads, never in chunk workers.
        import matplotlib.pyplot as plt
        from foxes.output import FlowPlots2D
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
            verbosity = algo.verbosity if self.verbosity is None else self.verbosity
            if verbosity > 0:
                print(f"{self.name}: Writing {path}")

            # Ensure renderer failures cannot leak Matplotlib figures.
            fig, ax = self._create_figure()
            try:
                animator = self._create_animator(algo, farm, fig, ax)
                for create_generator in extra_generators:
                    animator.add_generator(create_generator(ax))
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

    def _create_figure(self) -> tuple[Figure, Axes]:
        """Create the figure and axes used for the flow field."""
        import matplotlib.pyplot as plt

        return plt.subplots(figsize=self.figsize)

    def _create_animator(
        self, algo: Algorithm, farm: Dataset, fig: Figure, ax: Axes
    ) -> Animator:
        """Create the animator before adding its flow-field generator."""
        from foxes.output import FarmLayoutOutput

        if self._layout_pars is not None:
            FarmLayoutOutput(algo.farm).get_figure(fig=fig, ax=ax, **self._layout_pars)
        return Animator(fig=fig)

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


class FlowTurbineAnimation(FlowAnimation):
    """Write flow animations with a synchronized state-turbine trace panel.

    Each animated flow variable shares the same lower panel, using original
    farm turbine indices. Chunk/state selection, point evaluation, filenames
    and encoding follow ``FlowAnimation``.
    """

    def __init__(
        self,
        turbines: list[int] | None = None,
        turbine_variable: str = FV.REWS,
        layout_pars: dict[str, Any] | None = None,
        ref_xy: list[float] | tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the turbine panel and optional flow-field annotations.

        Parameters
        ----------
        turbines
            Original farm turbine indices to plot; ``None`` selects all.
            An explicit selection must be nonempty and contain valid indices.
        turbine_variable
            Farm variable with dimensions ``(FC.STATE, FC.TURBINE)`` to trace.
        layout_pars
            Options for ``FarmLayoutOutput.get_figure`` on the flow axes.
            ``None`` disables layout annotations; ``fig`` and ``ax`` are supplied
            by this output.
        ref_xy
            Finite reference point coordinates in metres, shown as a red cross.
            ``None`` disables the marker.
        kwargs
            Options for ``FlowAnimation``, including ``figsize``,
            ``grid_pars``, ``plot_pars``, frame selection and file paths.
            The upper and lower panels have a height ratio of three to one.
        """
        # Retain configuration until initialization prepares private snapshots.
        super().__init__(layout_pars=layout_pars, **kwargs)
        self.turbines = turbines
        self.turbine_variable = turbine_variable
        self.ref_xy = ref_xy

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Validate panel configuration before initializing flow dependencies.

        Parameters
        ----------
        algo
            Farm algorithm supplying the available turbine indices.
        loaded_data
            Shared initialization data to extend.
        force
            Reinitialize the output and its scientific dependencies.
        verbosity
            Initialization verbosity.

        Returns
        -------
        loaded_data
            Combined loaded model data, including inherited flow-grid data.

        Raises
        ------
        ValueError
            If the panel configuration is invalid or the output is running.
        """
        # Validate before retaining selections or building scientific submodels.
        if self.running:
            raise ValueError(f"{self.name}: Cannot initialize while running")
        if not self.initialized or force:
            turbines = (
                list(range(algo.n_turbines)) if self.turbines is None else self.turbines
            )
            if not turbines or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < algo.n_turbines
                for i in turbines
            ):
                raise ValueError(f"{self.name}: Invalid turbine selection {turbines}")
            if not isinstance(self.turbine_variable, str) or not self.turbine_variable:
                raise ValueError(f"{self.name}: turbine_variable must be nonempty")
            if self.ref_xy is not None and (
                np.shape(self.ref_xy) != (2,) or not np.all(np.isfinite(self.ref_xy))
            ):
                raise ValueError(
                    f"{self.name}: ref_xy must contain two finite coordinates"
                )

            # Snapshot caller-owned settings without retaining runtime buffers.
            self._turbines = tuple(turbines)
            self._turbine_variable = self.turbine_variable
            self._ref_xy = None if self.ref_xy is None else tuple(self.ref_xy)
        return super().initialize(algo, loaded_data, force, verbosity)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect flow fields and farm-order traces for selected states.

        Parameters
        ----------
        algo
            Running farm algorithm.
        mdata
            Chunk model data, including the inherited state selection.
        fdata
            Chunk farm data in original turbine order.
        tdata
            Supplied target data, or ``None`` for generated flow grids.
        model_results
            Scientific calculation results passed to the inherited collector.
        chunk_results
            Shared output payload mapping to extend.

        Returns
        -------
        chunk_results
            Private numerical animation payloads including the trace variable.

        Raises
        ------
        KeyError
            If the requested farm trace variable is unavailable.
        ValueError
            If the trace variable does not have state-turbine dimensions.
        """
        # Extend only selected payloads, keeping all model configuration read-only.
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        payload = results.get(self._result_key)
        if payload is not None:
            variable = self._turbine_variable
            if variable not in fdata:
                raise KeyError(f"{self.name}: Missing turbine variable '{variable}'")
            if fdata.dims[variable] != (FC.STATE, FC.TURBINE):
                raise ValueError(
                    f"{self.name}: '{variable}' must have state-turbine dimensions"
                )
            selection = (
                slice(None) if self._selection is None else mdata[self._selection]
            )
            payload["farm"][variable] = np.broadcast_to(
                fdata[variable], (fdata.n_states, fdata.n_turbines)
            )[selection].copy()
        return results

    def _create_figure(self) -> tuple[Figure, Axes]:
        """Create the flow axes above a smaller turbine-trace panel."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 1, figsize=self.figsize, gridspec_kw={"height_ratios": [3, 1]}
        )
        return fig, axes[0]

    def _create_animator(
        self, algo: Algorithm, farm: Dataset, fig: Figure, ax: Axes
    ) -> Animator:
        """Add synchronized turbine traces and optional flow annotations."""
        from foxes.output import FarmResultsEval

        # Attach traces to the figure, never to the running output instance.
        animator = super()._create_animator(algo, farm, fig, ax)
        animator.add_generator(
            FarmResultsEval(farm, algo=algo).gen_stdata(
                turbines=list(self._turbines),
                variable=self._turbine_variable,
                fig=fig,
                ax=fig.axes[1],
                ret_im=True,
                legloc="upper right",
                animated=True,
            )
        )

        # Annotate only the flow axes, leaving inherited scales and grids intact.
        if self._ref_xy is not None:
            ax.scatter(
                [self._ref_xy[0]],
                [self._ref_xy[1]],
                marker="x",
                color="red",
                s=80,
                animated=True,
            )
        return animator
