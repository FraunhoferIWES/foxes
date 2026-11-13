"""Collect flow chunks and save unified flow and turbine animations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, TData
from foxes.utils import Dict

from .animator import Animator
from .slice_data import SliceData

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from foxes.core import Algorithm
    from foxes.core.model import LoadedData


class FlowAnimation(SliceData):
    """Write one state-ordered planar flow animation per variable.

    Supplied target data is reused without further scientific calculations.
    Farm-only runs evaluate a shared grid in bounded point blocks. Workers
    return independent payloads; only finalization creates figures and files.
    """

    def _creates_figures(self) -> bool:
        """Identify animations as image-only, including their point grids."""
        return True

    def __init__(
        self,
        chunk: int | tuple[int, int] | list[int | tuple[int, int]] | None = None,
        variables: list[str] | None = None,
        plane: str = "xy",
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
        show: bool = False,
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
        plane
            Slice orientation: ``"xy"``, ``"xz"`` or ``"yz"``.
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
            Options for the selected plane's grid function, such as
            ``resolution``, ``n_img_points`` and plane-specific bounds, spacing
            and normal-axis position.
        plot_pars
            Flow-figure options, including ``vmin``, ``vmax``, ``cmap``,
            ``levels`` and ``quiver_n`` for rectangular grids. Other targets
            use Matplotlib ``scatter`` options, with
            ``vmin`` and ``vmax`` selecting the shared colour scale.
        ambient
            Skip wake calculations on generated grids. Supplied target data is
            always used as given, independently of this option.
        layout_pars
            ``FarmLayoutOutput`` figure options on XY flow axes. ``None``
            disables annotations; do not supply ``fig`` or ``ax``. Vertical
            planes do not display farm-layout annotations.
        verbosity
            Diagnostic verbosity; ``None`` uses ``algo.verbosity``. Level zero
            is silent; one announces creation; two adds writing and successful
            completion messages, frame counts and encoder settings; three
            reports each frame's composition and encoding progress. Success is
            announced only after the file has been written.
        show
            Play each completed animation after saving it, at ``fps`` frames per
            second. Display blocks until its window closes; the figure is then
            released. No animation or figure is retained on this model.
        kwargs
            Common output options, including ``out_dir`` and ``out_fname_fun``.
        """
        # Retain configuration only; initialization validates and prepares it.
        super().__init__(
            plane=plane,
            variables=variables,
            chunk=chunk,
            states_sel=states_sel,
            states_isel=states_isel,
            grid_pars=grid_pars,
            ambient=ambient,
            show=show,
            **kwargs,
        )
        self.base_name = base_name
        self.fps = fps
        self.figsize = figsize
        self.plot_pars = plot_pars
        self.layout_pars = layout_pars
        self.verbosity = verbosity

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Retain exact animation options for run-local rendering.

        Parameters
        ----------
        algo
            Executing algorithm.
        model_data
            Initial model dataset before engine subset selection.
        farm_data
            Initial or previously calculated farm dataset.
        point_data
            Initial target dataset, or ``None`` for farm calculations.
        extra_data
            Non-array model inputs.

        Returns
        -------
        output_results
            Slice inputs extended with private plot and layout options.
        """
        results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )
        dict.update(
            results,
            {
                self.var("plot_pars"): self.plot_pars,
                self.var("layout_pars"): self.layout_pars,
            },
        )
        return results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Stash animation options outside running worker models.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Lifecycle stash, or ``None`` to discard configuration.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["plot_pars"] = self.plot_pars
            data_stash[self.name]["layout_pars"] = self.layout_pars
        del self.plot_pars, self.layout_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore exact animation options after execution.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Lifecycle stash, or ``None`` for discarded configuration.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.plot_pars = data_stash[self.name].pop("plot_pars")
            self.layout_pars = data_stash[self.name].pop("layout_pars")

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
            if self.layout_pars is not None and {"fig", "ax"} & self.layout_pars.keys():
                raise ValueError(f"{self.name}: layout_pars must not supply fig or ax")

            # Resolve the animation path without copying parent-owned plot objects.
            self._base_name = Path(self.base_name)
        return super().initialize(algo, loaded_data, force, verbosity)

    def _required_variables(self) -> tuple[str, ...]:
        """Include vector fields required by frame rendering."""
        return tuple(dict.fromkeys([*super()._required_variables(), FV.WD, FV.WS]))

    def _farm_variables(self) -> tuple[str, ...]:
        """Retain farm geometry needed by state-resolved frame overlays."""
        return (FV.X, FV.Y, FV.H, FV.D, FV.YAW, FV.AMB_WD, FV.YAWM)

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
        chunks = output_results.get(self._result_key, {})
        chunks = self._selected_chunks(chunks, n_chunks_states, n_chunks_targets)

        # Assemble already selected numerical frames before rendering once.
        if chunks:
            payload = self._combine_chunks(chunks)
            self._render(
                algo,
                payload,
                output_results,
                output_results.get(self.var("extra_generators"), ()),
            )
        output_results.pop(self._result_key, None)

        # Consume inherited slice inputs and animation-specific render inputs.
        output_results.pop(self.var("write_pars"))
        output_results.pop(self.var("normalize"))
        output_results.pop(self.var("normalize_v"))
        output_results.pop(self.var("label_map"))
        output_results.pop(self.var("vmin"))
        output_results.pop(self.var("vmax"))
        output_results.pop(self.var("plot_pars"))
        output_results.pop(self.var("layout_pars"))
        output_results.pop(self.var("extra_generators"), None)

    def _render(
        self,
        algo: Algorithm,
        payload: dict[str, Any],
        output_results: Dict[str, Any],
        extra_generators: tuple[Callable[[Axes], Iterator[Any]], ...] = (),
    ) -> None:
        """Render and save collected fields, then optionally display them.

        Parameters
        ----------
        algo
            Algorithm supplying farm metadata and verbosity.
        payload
            Combined state-ordered fields, points and overlay geometry.
            Arrays are borrowed only for parent-side rendering.
        output_results
            Run-local results containing private rendering options and any
            turbine-panel selections, all absent from the running output model.
        extra_generators
            Factories adding synchronized artists to the flow axes.

        Notes
        -----
        ``nofig`` suppresses rendering. Figures close even if encoding or
        blocking display fails; no artists are retained on the model.
        """
        from ._flow_animation_render import render_flow_animation

        render_flow_animation(self, algo, payload, output_results, extra_generators)

    def _create_figure(self) -> tuple[Figure, Axes]:
        """Create a figure for the flow field without retaining it on the model.

        Returns
        -------
        figure
            New figure, owned and closed by the rendering caller.
        axes
            Axes used for flow-field artists.
        """
        import matplotlib.pyplot as plt

        return plt.subplots(figsize=self.figsize)

    def _create_animator(
        self,
        algo: Algorithm,
        farm: Dataset,
        fig: Figure,
        ax: Axes,
        output_results: Dict[str, Any],
    ) -> Animator:
        """Create a figure-local animator and optional static farm overlay.

        Parameters
        ----------
        algo
            Algorithm supplying farm geometry for the layout overlay.
        farm
            Selected state/turbine overlay fields, borrowed during rendering.
        fig
            Shared figure owned by the caller.
        ax
            Flow axes receiving any configured XY layout overlay.
        output_results
            Run-local results containing private layout options.

        Returns
        -------
        animator
            Local compositor with no flow-field generator added yet.
        """
        from ._layout import _get_figure
        from .farm_layout import FarmLayoutOutput

        layout_pars = output_results[self.var("layout_pars")]
        if layout_pars is not None and self.plane == "xy":
            _get_figure(
                algo.farm,
                farm,
                algo,
                FarmLayoutOutput(),
                fig=fig,
                ax=ax,
                **layout_pars,
            )
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
        """Generate scatter frames for irregular or moving target locations.

        Parameters
        ----------
        fig
            Shared figure, retained by the caller through encoding and display.
        ax
            Flow axes receiving artists with fixed coordinate/color limits.
        payload
            State-ordered point coordinates and scientific fields.
        variable
            Field to render without resampling.
        parameters
            Renderer parameters, including optional color-scale bounds.
        options
            Figure-local scatter options; ``vlabel`` is consumed for the colorbar.

        Yields
        ------
        frame
            ``(figure, artists)`` pairs containing a scatter collection and
            state-label text, in the payload's state order.
        """
        # Use fixed axes and a shared color scale so frames remain comparable.
        from matplotlib.colors import Normalize

        from .grids import project_points

        projected = project_points(
            payload["points"],
            self.plane,
            270 if self.plane == "xy" else self._grid_options.get("x_direction", 270),
        )
        indices = ["xyz".index(axis) for axis in self.plane]
        points, values = projected[..., indices], payload["values"][variable]
        norm = Normalize(
            np.min(values) if parameters["vmin"] is None else parameters["vmin"],
            np.max(values) if parameters["vmax"] is None else parameters["vmax"],
        )
        label = options.pop("vlabel")
        ax.set(xlabel=f"{self.plane[0]} [m]", ylabel=f"{self.plane[1]} [m]")
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
            ``FarmLayoutOutput`` figure options on the flow axes.
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

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Retain initialized turbine and reference-point snapshots.

        Parameters
        ----------
        algo
            Executing algorithm.
        model_data
            Initial model dataset before engine subset selection.
        farm_data
            Initial or previously calculated farm dataset.
        point_data
            Initial target dataset, or ``None`` for farm calculations.
        extra_data
            Non-array model inputs.

        Returns
        -------
        output_results
            Flow inputs extended with the private, immutable panel selections.
        """
        results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )
        dict.update(
            results,
            {
                self.var("_turbines"): self._turbines,
                self.var("_ref_xy"): self._ref_xy,
            },
        )
        return results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Stash raw panel settings and their initialized snapshots.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Lifecycle stash, or ``None`` to discard panel settings.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name].update(
                turbines=self.turbines,
                ref_xy=self.ref_xy,
                _turbines=self._turbines,
                _ref_xy=self._ref_xy,
            )
        del self.turbines, self.ref_xy, self._turbines, self._ref_xy

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore panel settings and snapshots after execution.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Lifecycle stash, or ``None`` for discarded configuration.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.turbines = data_stash[self.name].pop("turbines")
            self.ref_xy = data_stash[self.name].pop("ref_xy")
            self._turbines = data_stash[self.name].pop("_turbines")
            self._ref_xy = data_stash[self.name].pop("_ref_xy")

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Render flow and turbine panels, then consume private panel inputs.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total target chunks.
        output_results
            Run-local numerical buffers and private rendering inputs.
        """
        super().finalize_results(
            algo, n_chunks_states, n_chunks_targets, output_results
        )
        output_results.pop(self.var("_turbines"))
        output_results.pop(self.var("_ref_xy"))

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
        """Create flow axes above a synchronized turbine-trace panel.

        Returns
        -------
        figure
            New two-panel figure, owned and closed by the rendering caller.
        axes
            Upper flow axes; the lower trace axes remain in ``figure.axes[1]``.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 1, figsize=self.figsize, gridspec_kw={"height_ratios": [3, 1]}
        )
        return fig, axes[0]

    def _create_animator(
        self,
        algo: Algorithm,
        farm: Dataset,
        fig: Figure,
        ax: Axes,
        output_results: Dict[str, Any],
    ) -> Animator:
        """Add synchronized turbine traces and optional flow annotations.

        Parameters
        ----------
        algo
            Algorithm supplying turbine metadata and inherited layout context.
        farm
            State-ordered fields in original turbine order.
        fig
            Shared two-panel figure, owned by the rendering caller.
        ax
            Upper flow axes receiving layout and reference-point annotations.
        output_results
            Run-local results containing private layout options, turbine indices
            and the reference-point position.

        Returns
        -------
        animator
            Figure-local compositor with the lower turbine-trace generator
            registered before the flow-field generator.
        """
        from ._flow_animation_render import turbine_frames

        # Attach traces to the figure, never to the running output instance.
        animator = super()._create_animator(algo, farm, fig, ax, output_results)
        animator.add_generator(
            turbine_frames(
                farm,
                turbines=output_results[self.var("_turbines")],
                variable=self._turbine_variable,
                fig=fig,
                ax=fig.axes[1],
            )
        )

        # Annotate only the flow axes, leaving inherited scales and grids intact.
        ref_xy = output_results[self.var("_ref_xy")]
        if ref_xy is not None:
            ax.scatter(
                [ref_xy[0]],
                [ref_xy[1]],
                marker="x",
                color="red",
                s=80,
                animated=True,
            )
        return animator
