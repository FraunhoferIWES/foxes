# mypy: disable-error-code=arg-type

from __future__ import annotations

from foxes.algorithms.sequential import SequentialPlugin
from foxes.algorithms.sequential.sequential import Sequential
from foxes.models.wake_frames.seq_dynamic_wakes import SeqDynamicWakes
from matplotlib.axes import Axes
from typing import Any, Iterator
from xarray import Dataset


class SeqWakeDebugPlugin(SequentialPlugin):
    """
    Plugin for creating wake debug plots in animations

    Attributes
    ----------
    show_p
        Flag for showing wake points
    show_v
        Flag for showing wake vectors
    vpars
        Additional parameters for vector lines
    ppars
        Additional parameters for point scatter


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
        show_p
            Flag for showing wake points
        show_v
            Flag for showing wake vectors
        vpars
            Additional parameters for vector lines
        ppars
            Additional parameters for point scatter

        """
        super().__init__()
        self.show_p = show_p
        self.show_v = show_v

        self.vpars = dict(color="blue")
        self.vpars.update({} if vpars is None else vpars)

        self.ppars = dict(color="blue")
        self.ppars.update(ppars)

    def initialize(self, algo: Sequential) -> None:
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo
            The current sequential algorithm

        """
        super().initialize(algo)
        self._data: list[tuple[Any, Any, Any]] = []

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

        wframe = algo.wake_frame
        if not isinstance(wframe, SeqDynamicWakes):
            raise ValueError(
                f"Wake frame not of type SeqDynamicWakes, got {type(algo.wake_frame).__name__}"
            )

        counter = algo.counter
        assert counter is not None
        N = counter + 1
        assert wframe._dt is not None
        dt = wframe._dt[counter] if counter < len(wframe._dt) else wframe._dt[-1]

        assert wframe._traces_p is not None
        assert wframe._traces_v is not None
        assert self._data is not None
        self._data.append(
            (
                dt,
                wframe._traces_p[:N].copy(),
                wframe._traces_v[:N].copy(),
            )
        )

    def gen_images(self, ax: Axes) -> Iterator[tuple[Any, list[Any]]]:
        """

        Parameters
        ----------
        ax
            The plotting axis

        Yields
        ------
        imgs
            The (figure, artists) tuple

        """
        assert self._data is not None
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
