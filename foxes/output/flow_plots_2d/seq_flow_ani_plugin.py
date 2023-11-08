from foxes.algorithms.sequential import SequentialIterPlugin
import matplotlib.pyplot as plt

from .flow_plots import FlowPlots2D

class SeqFlowAnimationPlugin(SequentialIterPlugin):
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

    def __init__(
            self, 
            orientation,
            runner=None,
            **pars
        ):
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
        #self.pars["animated"] = True

    def initialize(self, iter):
        """
        Initialize data based on the intial iterator
        
        Parameters
        ----------
        iter: foxes.algorithms.sequential.models.SequentialIter
            The initialized iterator
        
        """
        super().initialize(iter)
        self._data = []

    def update(self, iter, fres, pres=None):
        """
        Updates data based on current iteration
        
        Parameters
        ----------
        iter: foxes.algorithms.sequential.models.SequentialIter
            The latest iterator
        fres: xarray.Dataset
            The current farm results
        pres: xarray.Dataset, optional
            The current point results
        
        """
        super().update(iter, fres, pres)

        o = FlowPlots2D(iter.algo, fres, self.runner)

        if self.orientation == "xy":
            self._data.append(next(o.gen_states_fig_xy(**self.pars)))
        elif self.orientation == "xz":
            self._data.append(next(o.gen_states_fig_xz(**self.pars)))
        elif self.orientation == "yz":
            self._data.append(next(o.gen_states_fig_yz(**self.pars)))
        else:
            raise KeyError(f"Unkown orientation '{self.orientation}', choises: xy, xz, yz")
    
    def gen_images(self):
        for d in self._data:
            yield d
