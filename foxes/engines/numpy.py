from __future__ import annotations

from typing import Any

from .threads import ThreadsEngine, ThreadsEngineRunner


class NumpyEngine(ThreadsEngine):
    """Execute NumPy chunks synchronously in the calling process."""

    def _create_pool(self) -> None:
        """Synchronous execution needs no worker resources."""

    def _shutdown_pool(self) -> None:
        """Synchronous execution has no worker resources to release."""

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a chunk immediately and return its completed result.

        Parameters
        ----------
        f
            Function called synchronously as ``f(*args, **kwargs)``.
        args
            Arguments for the function
        kwargs
            Arguments for the function

        Returns
        -------
        future
            Completed-result mapping consumed by the engine's collection loop.

        """
        return {"result": f(*args, **kwargs), "done": True}

    def await_result(self, future: dict[str, Any]) -> Any:
        """
        Waits for result from a future

        Parameters
        ----------
        future
            The future

        Returns
        -------
        result
            The calculation result

        """
        return future["result"]

    def future_is_done(self, future: dict[str, Any]) -> bool:
        """
        Checks if a future is done

        Parameters
        ----------
        future
            The future

        Returns
        -------
        is_done
            True if the future is done

        """
        return future["done"]

    def map(
        self,
        func: Any,
        inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Runs a function on a list of files

        Parameters
        ----------
        func
            Function to be called on each file,
            func(input, *args, **kwargs) -> data
        inputs
            The input data list
        args
            Arguments for func
        kwargs
            Keyword arguments for func

        Returns
        -------
        results
            Results for the submitted inputs

        """
        return [func(input, *args, **kwargs) for input in inputs]

    def _get_execution_description(self) -> str:
        """Describe synchronous chunk execution for calculation progress.

        Returns
        -------
        description
            Loop-based execution without worker resources.
        """
        return "a loop over chunks"

    def get_start_calc_message(
        self,
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> str:
        """Helper function for start calculation message"""
        msg = f"{self.name}: Starting calculation using a loop over"
        msg += f" {n_chunks_states} states chunks"
        if n_chunks_targets > 1:
            msg += f" and {n_chunks_targets} targets chunks"
        msg += "."
        return msg

    def new_runner(self) -> ThreadsEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner
            The engine runner

        """
        return ThreadsEngineRunner()
