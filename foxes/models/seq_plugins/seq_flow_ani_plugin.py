# mypy: disable-error-code=arg-type
# mypy: disable-error-code=assignment

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from matplotlib.axes import Axes
from xarray import Dataset

import foxes.variables as FV
from foxes.algorithms.sequential import SequentialPlugin
from foxes.algorithms.sequential.sequential import Sequential
from foxes.models.outputs import FlowPlots2D, _flow_figures


class SeqFlowAnimationPlugin(SequentialPlugin):
    """An iterator plugin buffering plane data for later flow-animation frames."""

    def __init__(
        self,
        orientation: str,
        title_fun: Callable[[int, Any], str] | None = None,
        var: str = FV.WS,
        grid_pars: dict[str, Any] | None = None,
        **data_pars: Any,
    ) -> None:
        """Configure the animation plane, plotted field and grid evaluation.

        Parameters
        ----------
        orientation
            Plane orientation: ``"xy"``, ``"xz"`` or ``"yz"``.
        title_fun
            Callback ``title_fun(iteration, state_label)`` producing a frame
            title during each update. ``None`` allows a fixed plotting title.
        var
            Physical point field to plot, in its established ``FV`` units.
        grid_pars
            Grid resolution or point count, bounds and fixed plane coordinate
            in metres. The mapping is copied.
        data_pars
            Additional ``FlowPlots2D`` constructor options for numerical data.

        Raises
        ------
        ValueError
            If ``orientation`` is not one of the supported planes.
        """
        super().__init__()
        self.orientation = orientation
        self.data_pars = data_pars
        self._tfun = title_fun
        self.variable = var
        self.grid_pars = {} if grid_pars is None else dict(grid_pars)
        if orientation not in ("xy", "xz", "yz"):
            raise ValueError(
                f"Unknown orientation '{orientation}'; choose xy, xz or yz"
            )

    def initialize(self, algo: Sequential) -> None:
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo
            The current sequential algorithm

        """
        super().initialize(algo)
        self._data: list[tuple[Any, list[Any]]] = []
        self._titles: list[str] = []

    def update(
        self, algo: Sequential, fres: Dataset, pres: Dataset | None = None
    ) -> None:
        """
        Buffer one state's numerical plane data and optional frame title.

        Parameters
        ----------
        algo
            Active sequential iterator, supplying wake history for this step.
        fres
            Current farm fields on ``(FC.STATE, FC.TURBINE)`` in original farm
            order. The dataset is retained for later frame overlays, not copied.
        pres
            Current iterator point results, if available; plane data are
            calculated independently through ``algo.calc_outputs``.

        Notes
        -----
        The plugin owns buffered arrays and titles. Shared plane metadata is
        retained only for the first frame; later frames reuse it during image
        generation. Output models remain configuration-only.

        """
        super().update(algo, fres, pres)

        output = FlowPlots2D(
            var=self.variable,
            plane=self.orientation,
            reduction="states",
            plot=False,
            data_format="numpy",
            grid_pars=self.grid_pars,
            **self.data_pars,
        )

        if self._tfun is not None:
            assert algo.counter is not None
            self._titles.append(self._tfun(algo.counter, algo.states.index()[0]))

        d = algo.calc_outputs(outputs=output, farm_results=fres, finalize=False)

        # minimize stored data:
        d = list(d)
        if len(self._data) > 0:
            d[0] = None
            d[-1] = None

        self._data.append((fres, d))

    def gen_images(self, ax: Axes, **plot_pars: Any) -> Iterator[tuple[Any, Any]]:
        """Yield buffered plane frames and consume their stored numerical data.

        Parameters
        ----------
        ax
            Existing plotting axes, reused for every yielded frame. The caller
            owns its figure and controls display and saving.
        plot_pars
            Plane-renderer options, including colour limits and a fixed
            ``title`` only when no ``title_fun`` was configured.

        Yields
        ------
        frame
            ``(figure, artists)`` pair in iterator-state order. Fields and
            overlays use the configured plane and original farm turbine order.

        Notes
        -----
        Each advanced frame is removed from the plugin's buffers. Stopping
        iteration early leaves unconsumed frames available for a later call.

        """
        add_bar = (
            plot_pars.get("vmin", None) is None or plot_pars.get("vmax", None) is None
        )

        fig = ax.get_figure()
        parameters = None
        gdata = None
        while len(self._data):
            farm_results, d = self._data.pop(0)

            if "title" in plot_pars:
                assert self._tfun is None, (
                    "Cannot have a title function together with the 'title' parameter"
                )
                plot_pars = plot_pars.copy()
                title = plot_pars.pop("title")
            elif self._tfun is not None:
                title = self._titles.pop(0)
            else:
                title = None

            if d[-1] is not None:
                parameters = d[0]
                gdata = d[-1]
            else:
                d[0] = parameters
                d[-1] = gdata

            yield next(
                getattr(_flow_figures, f"_gen_states_fig_{self.orientation}")(
                    farm_results,
                    FlowPlots2D(plane=self.orientation),
                    d,
                    ax=ax,
                    fig=fig,
                    title=title,
                    add_bar=add_bar,
                    ret_im=True,
                    animated=True,
                    **plot_pars,
                )
            )
