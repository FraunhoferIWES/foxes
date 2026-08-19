# mypy: disable-error-code=arg-type
# mypy: disable-error-code=assignment

from __future__ import annotations

from foxes.algorithms.sequential import SequentialPlugin
from foxes.algorithms.sequential.sequential import Sequential
from matplotlib.axes import Axes
from typing import Any, Callable, Iterator
from xarray import Dataset

from ..flow_plots_2d.flow_plots import FlowPlots2D


class SeqFlowAnimationPlugin(SequentialPlugin):
    """
    Plugin for creating data for a 2D flow animation
    during sequential iterations

    Attributes
    ----------
    orientation
        The orientation, either "yx", "xz" or "yz"
    title_fun
        A function that takes the current iteration and state index
        and returns a title string.
    data_pars
        Additional parameters for plot data calculation


    """

    def __init__(
        self,
        orientation: str,
        title_fun: Callable[[int, Any], str] | None = None,
        **data_pars: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        orientation
            The orientation, either "yx", "xz" or "yz"
        title_fun
            A function that takes the current iteration and state index
            and returns a title string.
        data_pars
            Additional parameters for plot data calculation


        """
        super().__init__()
        self.orientation = orientation
        self.data_pars = data_pars
        self._tfun = title_fun

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
        Updates data based on current iteration

        Parameters
        ----------
        algo
            The latest sequential algorithm
        fres
            The latest farm results
        pres
            The latest point results

        """
        super().update(algo, fres, pres)

        o = FlowPlots2D(algo, fres)

        if self._tfun is not None:
            assert algo.counter is not None
            self._titles.append(self._tfun(algo.counter, algo.states.index()[0]))

        d: Any
        if self.orientation == "xy":
            d = o.get_states_data_xy(**self.data_pars, data_format="numpy")
        elif self.orientation == "xz":
            d = o.get_states_data_xz(**self.data_pars, data_format="numpy")
        elif self.orientation == "yz":
            d = o.get_states_data_yz(**self.data_pars, data_format="numpy")
        else:
            raise KeyError(
                f"Unkown orientation '{self.orientation}', choises: xy, xz, yz"
            )

        # minimize stored data:
        d = list(d)
        if len(self._data) > 0:
            d[0] = None
            d[-1] = None

        self._data.append((o, d))

    def gen_images(self, ax: Axes, **plot_pars: Any) -> Iterator[tuple[Any, Any]]:
        """

        Parameters
        ----------
        ax
            The plotting axis
        plot_pars
            Additional parameters for plotting

        Yields
        ------
        imgs
            The (figure, artists) tuple

        """
        add_bar = (
            plot_pars.get("vmin", None) is None or plot_pars.get("vmax", None) is None
        )

        fig = ax.get_figure()
        parameters = None
        gdata = None
        while len(self._data):
            o, d = self._data.pop(0)

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
                o.gen_states_fig_xy(
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
