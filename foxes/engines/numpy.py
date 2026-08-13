from .threads import ThreadsEngine, ThreadsEngineRunner


class NumpyEngine(ThreadsEngine):
    """
    The numpy engine for foxes calculations.

    :group: engines

    """

    def submit(self, f, *args, **kwargs):
        """
        Submits a job to worker, obtaining a future

        Parameters
        ----------
        f: Callable
            The function f(*args, **kwargs) to be
            submitted
        args: tuple, optional
            Arguments for the function
        kwargs: dict, optional
            Arguments for the function

        Returns
        -------
        future: object
            The future object

        """
        return {"f": f, "args": args, "kwargs": kwargs, "result": None, "done": False}

    def await_result(self, future):
        """
        Waits for result from a future

        Parameters
        ----------
        future: object
            The future

        Returns
        -------
        result: object
            The calculation result

        """
        if not future["done"]:
            f, args, kwargs = future.pop("f"), future.pop("args"), future.pop("kwargs")
            future["result"] = f(*args, **kwargs)
            future["done"] = True

        return future["result"]

    def future_is_done(self, future):
        """
        Checks if a future is done

        Parameters
        ----------
        future: object
            The future

        Returns
        -------
        is_done: bool
            True if the future is done

        """
        return future["done"]

    def map(
        self,
        func,
        inputs,
        *args,
        **kwargs,
    ):
        """
        Runs a function on a list of files

        Parameters
        ----------
        func: Callable
            Function to be called on each file,
            func(input, *args, **kwargs) -> data
        inputs: array-like
            The input data list
        args: tuple, optional
            Arguments for func
        kwargs: dict, optional
            Keyword arguments for func

        Returns
        -------
        results: list
            The list of results

        """
        return [func(input, *args, **kwargs) for input in inputs]

    def get_start_calc_message(
        self,
        n_chunks_states,
        n_chunks_targets,
    ):
        """Helper function for start calculation message"""
        msg = f"{self.name}: Starting calculation using a loop over"
        msg += f" {n_chunks_states} states chunks"
        if n_chunks_targets > 1:
            msg += f" and {n_chunks_targets} targets chunks"
        msg += "."
        return msg

    def new_runner(self):
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return ThreadsEngineRunner()
