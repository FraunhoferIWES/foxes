from __future__ import annotations

from foxes.algorithms.sequential import SequentialPlugin
from foxes.models.wake_frames.seq_dynamic_wakes import SeqDynamicWakes
from typing import Any, Iterator


class SeqWakeDebugPlugin(SequentialPlugin):
    """
    Plugin for creating wake debug plots in animations

    Attributes
    ----------
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

    def __init__(
        self,
        show_p: bool = True,
        show_v: bool = True,
        vpars: dict[str, Any] | None = None,
        **ppars: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
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
        self.show_p = show_p
        self.show_v = show_v

        self.vpars = dict(color="blue")
        self.vpars.update({} if vpars is None else vpars)

        self.ppars = dict(color="blue")
        self.ppars.update(ppars)

    def initialize(self, algo) -> None:
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The current sequential algorithm

        """
        super().initialize(algo)
        self._data: list[tuple[Any, Any, Any]] = []

    def update(self, algo, fres, pres=None) -> None:
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

        wframe = algo.wake_frame
        if not isinstance(wframe, SeqDynamicWakes):
            raise ValueError(
                f"Wake frame not of type SeqDynamicWakes, got {type(algo.wake_frame).__name__}"
            )

        counter = algo.counter
        N = counter + 1
        dt = wframe._dt[counter] if counter < len(wframe._dt) else wframe._dt[-1]

        self._data.append(
            (
                dt,
                wframe._traces_p[:N].copy(),
                wframe._traces_v[:N].copy(),
            )
        )

    def gen_images(self, ax) -> Iterator[tuple[Any, list[Any]]]:
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
        while len(self._data):
            dt, pts, v = self._data.pop(0)

            N = len(pts)
            artists: list[Any] = []
            assert self.algo is not None
            if self.show_p:
                artists += [
                    ax.scatter(
                        pts[:, downwind_index, 0],
                        pts[:, downwind_index, 1],
                        animated=True,
                        **self.ppars,
                    )
                    for downwind_index in range(self.algo.n_turbines)
                ]

            if self.show_v:
                for downwind_index in range(self.algo.n_turbines):
                    for i in range(N):
                        p = pts[i, downwind_index]
                        dxy = v[i, downwind_index] * dt
                        artists.append(
                            ax.arrow(
                                p[0],
                                p[1],
                                dxy[0],
                                dxy[1],
                                length_includes_head=True,
                                animated=True,
                                **self.vpars,
                            )
                        )

            yield ax.get_figure(), artists
