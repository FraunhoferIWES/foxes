import matplotlib.animation as animation


class Animator:
    """
    Creates an animation from generators
    that yield lists of artists.

    Attributes
    ----------
    fig: pyplot.Figure
        The figure object

    :group: output

    """

    def __init__(self, fig=None):
        """
        Constructor.

        Parameters
        ----------
        fig: pyplot.Figure, optional
            The figure object

        """
        self.fig = fig
        self._gens = []

    def add_generator(self, gen):
        """
        Add a generator.

        Parameters
        ----------
        gen: Generator
            A generator that yields (fig, list of Artist)

        """
        self._gens.append(gen)

    @property
    def generators(self):
        """
        The artist generators

        Returns
        -------
        gens: list of generators
            Generators that yield (fig, list of Artist)

        """
        return self._gens

    def animate(self, verbosity=1, **kwargs):
        """
        Create the animation

        Parameters
        ----------
        verbostiy: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for pyplot.animation.ArtistAnimation

        Returns
        -------
        ani: pyplot.animation.ArtistAnimation
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
