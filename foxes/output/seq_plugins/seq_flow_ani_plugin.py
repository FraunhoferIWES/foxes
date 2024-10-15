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
            The current sequetial algorithm

        """
        super().initialize(algo)
        self._data = []

    def update(self, algo, fres, pres=None):
        """
        Updates data based on current iteration

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The latest sequetial algorithm
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
            
        self._data.append((fres, d))

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
        for fres, d in self._data:
            
            o = FlowPlots2D(self.algo, fres)
            yield next(o.gen_states_fig_xy(**self.pars, ax=ax, fig=fig, ret_im=True, precalc=d))
            
            if (
                self.pars.get("vmin", None) is not None
                and self.pars.get("vmax", None) is not None
            ):
                self.pars["add_bar"] = False
