from __future__ import annotations

from typing import Any

from .threads import ThreadsEngine, ThreadsEngineRunner


class NumpyEngine(ThreadsEngine):
    """
    The numpy engine for foxes calculations.
    """

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Submits a job to worker, obtaining a future

        Parameters
        ----------
        f
            The function f(*args, **kwargs) to be
            submitted
        args
            Arguments for the function
        kwargs
            Arguments for the function

        Returns
        -------
        future
            The future object

        """
        return {"f": f, "args": args, "kwargs": kwargs, "result": None, "done": False}

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
        if not future["done"]:
            f, args, kwargs = future.pop("f"), future.pop("args"), future.pop("kwargs")
            future["result"] = f(*args, **kwargs)
            future["done"] = True

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
