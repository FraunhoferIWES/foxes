from __future__ import annotations

import matplotlib.animation as animation
from matplotlib.figure import Figure
from typing import Any


class Animator:
    """
    Creates an animation from generators
    that yield lists of artists.
    """

    def __init__(self, fig: Figure | None = None) -> None:
        """
        Parameters
        ----------
        fig
            The figure object
        """
        self.fig = fig
        self._gens: list[Any] = []

    def add_generator(self, gen: Any) -> None:
        """
        Add a generator.

        Parameters
        ----------
        gen
            A generator that yields a figure and artist collection

        """
        self._gens.append(gen)

    @property
    def generators(self) -> list[Any]:
        """
        The artist generators

        Returns
        -------
        gens
            Generators that yield a figure and artist collection

        """
        return self._gens

    def animate(self, verbosity: int = 1, **kwargs: Any) -> Any:
        """
        Create the animation

        Parameters
        ----------
        verbostiy
            The verbosity level, 0 = silent
        kwargs
            Arguments for pyplot.animation.ArtistAnimation

        Returns
        -------
        ani
            The animation

        """
        if len(self.generators) == 0:
            return None

        if verbosity > 0:
            print("Creating animation data")

        si = 0
        arts = []
        while True:
            if verbosity > 1:
                print(f"  Frame {si}")

            harts = []
            for g in self.generators:
                try:
                    y = next(g)

                    if len(y) != 2:
                        raise ValueError(
                            f"Expecting yield (fig, artists) from generator {g}"
                        )

                    fig, artists = y
                    if self.fig is None:
                        self.fig = fig
                    elif fig is not self.fig:
                        raise ValueError(f"Wrong figure returned by generator {g}")

                    harts += [a for a in artists]

                except StopIteration:
                    pass

            if len(harts):
                arts.append(harts)
                si += 1
            else:
                break

        if verbosity > 1:
            print("Done.")

        kwa = dict(interval=200, blit=True, repeat_delay=2000)
        kwa.update(kwargs)
        ani = animation.ArtistAnimation(fig, arts, **kwa)

        return ani
