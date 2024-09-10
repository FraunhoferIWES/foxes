from foxes.algorithms.sequential import SequentialPlugin
from foxes.models.wake_frames.seq_dynamic_wakes import SeqDynamicWakes


class SeqWakeDebugPlugin(SequentialPlugin):
    """
    Plugin for creating wake debug plots in animations

    Attributes
    ----------
    ax: matplotlib.Axis
        The plotting axis
    show_p: bool
        Flag for showing wake points
    show_v: bool
        Flag for showing wake vectors
    vpars: dict
        Additional parameters for vector lines
    ppars: dict
        Additional parameters for point scatter

    :group: output.seq_plugins

    """

    def __init__(self, ax, show_p=True, show_v=True, vpars={}, **ppars):
        """
        Constructor.

        Parameters
        ----------
        ax: matplotlib.Axis
            The plotting axis
        show_p: bool
            Flag for showing wake points
        show_v: bool
            Flag for showing wake vectors
        vpars: dict
            Additional parameters for vector lines
        ppars: dict, optional
            Additional parameters for point scatter

        """
        super().__init__()
        self.ax = ax
        self.show_p = show_p
        self.show_v = show_v

        self.vpars = dict(color="blue")
        self.vpars.update(vpars)

        self.ppars = dict(color="blue")
        self.ppars.update(ppars)

    @property
    def fig(self):
        """
        The figure object

        Returns
        -------
        f: matplotlib.Figure
            The figure object

        """
        return self.ax.get_figure()

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

        wframe = algo.wake_frame
        if not isinstance(wframe, SeqDynamicWakes):
            raise ValueError(
                f"Wake frame not of type SeqDynamicWakes, got {type(algo.wake_frame).__name__}"
            )

        counter = algo.counter
        N = counter + 1
        pts = wframe._traces_p[:N]

        artists = []
        if self.show_p:
            artists += [
                self.ax.scatter(
                    pts[:, downwind_index, 0],
                    pts[:, downwind_index, 1],
                    animated=True,
                    **self.ppars,
                )
                for downwind_index in range(algo.n_turbines)
            ]

        if self.show_v:
            dt = wframe._dt[counter] if counter < len(wframe._dt) else wframe._dt[-1]
            for downwind_index in range(algo.n_turbines):
                for i in range(N):
                    p = pts[i, downwind_index]
                    dxy = wframe._traces_v[i, downwind_index] * dt
                    artists.append(
                        self.ax.arrow(
                            p[0],
                            p[1],
                            dxy[0],
                            dxy[1],
                            length_includes_head=True,
                            animated=True,
                            **self.vpars,
                        )
                    )

        self._data.append((self.fig, artists))

    def gen_images(self):
        for d in self._data:
            yield d
