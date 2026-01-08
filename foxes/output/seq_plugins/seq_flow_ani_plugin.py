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
    plot_pars: dict
        Additional parameters for plotting
    data_pars: dict
        Additional parameters for plot data calculation

    :group: output.seq_plugins

    """

    def __init__(self, orientation, data_pars={}, plot_pars={}):
        """
        Constructor.

        Parameters
        ----------
        orientation: str
            The orientation, either "yx", "xz" or "yz"
        plot_pars: dict
            Additional parameters for plotting
        data_pars: dict, optional
            Additional parameters for plot data calculation


        """
        super().__init__()
        self.orientation = orientation
        self.plot_pars = plot_pars
        self.data_pars = data_pars

        if "title" in self.plot_pars and callable(self.plot_pars["title"]):
            self._tfun = self.plot_pars.pop("title")
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
            self.plot_pars["title"] = self._tfun(
                algo.states.counter, algo.states.index()[0]
            )

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

        of = (
            fres
            if (
                "rotor_color" in self.plot_pars
                and self.plot_pars["rotor_color"] is not None
            )
            else None
        )

        self._data.append((of, d))

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
        parameters = None
        gdata = None
        while len(self._data):
            fres, d = self._data.pop(0)

            if d[-1] is not None:
                parameters = d[0]
                gdata = d[-1]
            else:
                d[0] = parameters
                d[-1] = gdata

            o = FlowPlots2D(self.algo, fres)

            yield next(
                o.gen_states_fig_xy(
                    d,
                    ax=ax,
                    fig=fig,
                    ret_im=True,
                    **self.plot_pars,
                )
            )

            del o, fres, d

            if (
                self.plot_pars.get("vmin", None) is not None
                and self.plot_pars.get("vmax", None) is not None
            ):
                self.plot_pars["add_bar"] = False
