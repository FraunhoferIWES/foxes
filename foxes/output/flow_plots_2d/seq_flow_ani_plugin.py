from foxes.algorithms.sequential import SequentialPlugin

from .flow_plots import FlowPlots2D


class SeqFlowAnimationPlugin(SequentialPlugin):
    """
    Plugin for creating data for a 2D flow animation
    during sequential iterations

    Attributes
    ----------
    orientation: str
        The orientation, either "yx", "xz" or "yz"
    runner: foxes.utils.runners.Runner
        The runner
    pars: dict
        Additional parameters for plotting

    :group: output.flow_plots_2d

    """

    def __init__(self, orientation, runner=None, **pars):
        """
        Constructor.

        Parameters
        ----------
        orientation: str
            The orientation, either "yx", "xz" or "yz"
        runner: foxes.utils.runners.Runner, optional
            The runner
        pars: dict, optional
            Additional parameters for plotting

        """
        super().__init__()
        self.orientation = orientation
        self.runner = runner
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

        o = FlowPlots2D(algo, fres, self.runner)

        if self._tfun is not None:
            self.pars["title"] = self._tfun(algo.states.counter, algo.states.index()[0])

        if self.orientation == "xy":
            self._data.append(next(o.gen_states_fig_xy(**self.pars)))
        elif self.orientation == "xz":
            self._data.append(next(o.gen_states_fig_xz(**self.pars)))
        elif self.orientation == "yz":
            self._data.append(next(o.gen_states_fig_yz(**self.pars)))
        else:
            raise KeyError(
                f"Unkown orientation '{self.orientation}', choises: xy, xz, yz"
            )

        if (
            self.pars.get("vmin", None) is not None
            and self.pars.get("vmax", None) is not None
        ):
            self.pars["add_bar"] = False

    def gen_images(self):
        for d in self._data:
            yield d
