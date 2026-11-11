"""Synchronized composition of Matplotlib artist-frame generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class Animator:
    """Synchronized animation composition from figure/artist generators."""

    def __init__(self, fig: Figure | None = None) -> None:
        """Configure the shared figure.

        Parameters
        ----------
        fig
            Shared figure, or ``None`` to use the first yielded figure.
        """
        self.fig = fig
        self._gens: list[Any] = []

    def add_generator(self, gen: Any) -> None:
        """Add a synchronized artist generator.

        Parameters
        ----------
        gen
            Generator yielding ``(figure, artists)`` pairs.
        """
        self._gens.append(gen)

    @property
    def generators(self) -> list[Any]:
        """The synchronized artist generators.

        Returns
        -------
        generators
            Generators yielding figures and artist collections.
        """
        return self._gens

    def animate(self, verbosity: int = 1, **kwargs: Any) -> Any:
        """Create a Matplotlib artist animation.

        Parameters
        ----------
        verbosity
            Diagnostic verbosity; zero is silent.
        kwargs
            Options for ``matplotlib.animation.ArtistAnimation``.

        Returns
        -------
        animation
            Composed animation, or ``None`` when no generators were added.

        Raises
        ------
        ValueError
            If a generator yields an invalid pair or a different figure.
        """
        from matplotlib.animation import ArtistAnimation

        if len(self.generators) == 0:
            return None

        if verbosity > 0:
            print("Creating animation data")

        # Consume generators in lockstep until no artists remain.
        frame_index = 0
        frames = []
        while True:
            if verbosity > 1:
                print(f"  Frame {frame_index}")

            frame_artists = []
            for generator in self.generators:
                try:
                    frame = next(generator)

                    if len(frame) != 2:
                        raise ValueError(
                            f"Expecting yield (fig, artists) from generator {generator}"
                        )

                    fig, artists = frame
                    if self.fig is None:
                        self.fig = fig
                    elif fig is not self.fig:
                        raise ValueError(
                            f"Wrong figure returned by generator {generator}"
                        )

                    frame_artists.extend(artists)

                except StopIteration:
                    pass

            if frame_artists:
                frames.append(frame_artists)
                frame_index += 1
            else:
                break

        if verbosity > 1:
            print("Done.")

        # Retain established playback defaults unless explicitly overridden.
        options = dict(interval=200, blit=True, repeat_delay=2000)
        options.update(kwargs)
        return ArtistAnimation(fig, frames, **options)
