"""Synchronous execution using the same output lifecycle as parallel engines."""

from __future__ import annotations


import numpy as np

from .numpy import NumpyEngine
from .threads import ThreadsEngineRunner


class SingleChunkEngineRunner(ThreadsEngineRunner):
    """Prepared-data output runner for synchronous single-chunk execution."""


class SingleChunkEngine(NumpyEngine):
    """Execute the full state/target domain synchronously in one chunk."""

    def __init__(
        self,
        chunk_size_states: int | None = None,
        chunk_size_points: int | None = None,
        n_procs: int = 1,
        progress_bar: bool | None = True,
        verbosity: int = 1,
    ) -> None:
        """Configure synchronous execution.

        Parameters
        ----------
        chunk_size_states
            Ignored; all states form one chunk.
        chunk_size_points
            Ignored; all targets form one chunk.
        n_procs
            Ignored; calculation uses the calling process.
        progress_bar
            Automatic interactive progress when true, log lines when false,
            or no progress when ``None``.
        verbosity
            Verbosity level; zero suppresses progress and diagnostics.
        """
        # Use the shared orchestration without creating a worker pool.
        super().__init__(n_procs=1, progress_bar=progress_bar, verbosity=verbosity)

    def calc_chunk_sizes(
        self, n_states: int, n_targets: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one chunk covering all states and targets.

        Parameters
        ----------
        n_states
            Number of selected states.
        n_targets
            Number of selected targets.

        Returns
        -------
        state_sizes
            One-element array containing the state count.
        target_sizes
            One-element array containing the target count.
        """
        # Preserve the single-chunk guarantee regardless of worker heuristics.
        return np.array([n_states]), np.array([n_targets])

    def new_runner(self) -> SingleChunkEngineRunner:
        """Create a synchronous prepared-data runner.

        Returns
        -------
        runner
            Runner using the same output hooks as thread and process engines.
        """
        # Keep the distinct public runner type for engine introspection.
        return SingleChunkEngineRunner()
