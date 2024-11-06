from copy import deepcopy

from foxes.algorithms.sequential import SequentialPlugin

from ..flow_plots_2d.flow_plots import FlowPlots2D


class SeqFlowAnimationPlugin(SequentialPlugin):
    """
    Plugin for creating data for a 2D flow animation
    during sequential iterations

    Attributes
    ----------
    orientation: str
        The orientation, either "yx", "xz" or "yz"
    pars: dict
        Additional parameters for plotting

    :group: output.seq_plugins

    """

    def __init__(self, orientation, **pars):
        """
        Constructor.

        Parameters
        ----------
        orientation: str
            The orientation, either "yx", "xz" or "yz"
        pars: dict, optional
            Additional parameters for plotting

        """
        super().__init__()
        self.orientation = orientation
        self.pars = pars

        if "title" in self.pars and callable(self.pars["title"]):
            self._tfun = self.pars.pop("title")
        else:
            self._tfun = None

    def initialize(self, algo):
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The current sequential algorithm

        """
        super().initialize(algo)
        self._data = []

    def update(self, algo, fres, pres=None):
        """
        Updates data based on current iteration

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The latest sequential algorithm
        fres: xarray.Dataset
            The latest farm results
        pres: xarray.Dataset, optional
            The latest point results

        """
        super().update(algo, fres, pres)

        o = FlowPlots2D(algo, fres)

        if self._tfun is not None:
            self.pars["title"] = self._tfun(algo.states.counter, algo.states.index()[0])

        if self.orientation == "xy":
            d = next(o.gen_states_fig_xy(**self.pars, precalc=True))
        elif self.orientation == "xz":
            d = next(o.gen_states_fig_xz(**self.pars, precalc=True))
        elif self.orientation == "yz":
            d = next(o.gen_states_fig_yz(**self.pars, precalc=True))
        else:
            raise KeyError(
                f"Unkown orientation '{self.orientation}', choises: xy, xz, yz"
            )

        # minimize stored data:
        od = [d[0], d[1], None]
        if len(self._data) == 0:
            od[2] = d[2]
        of = (
            fres
            if ("rotor_color" in self.pars and self.pars["rotor_color"] is not None)
            else None
        )

        self._data.append((of, od))

    def gen_images(self, ax):
        """

        Parameters
        ----------
        ax: matplotlib.Axis
            The plotting axis

        Yields
        ------
        imgs: tuple
            The (figure, artists) tuple

        """
        fig = ax.get_figure()
        gdata = None
        while len(self._data):

            fres, d = self._data.pop(0)

            if d[2] is not None:
                gdata = d[2]

            o = FlowPlots2D(self.algo, fres)

            yield next(
                o.gen_states_fig_xy(
                    **self.pars,
                    ax=ax,
                    fig=fig,
                    ret_im=True,
                    precalc=(d[0], d[1], gdata),
                )
            )

            del o, fres, d

            if (
                self.pars.get("vmin", None) is not None
                and self.pars.get("vmax", None) is not None
            ):
                self.pars["add_bar"] = False
