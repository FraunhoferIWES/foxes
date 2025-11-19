from .pool import PoolEngine


class NumpyEngine(PoolEngine):
    """
    The Numpy engine for foxes calculations.

    This engine runs everything in the main process, using numpy
    for vectorized calculations.

    :group: engines

    """

    def _create_pool(self):
        """Creates the pool"""
        pass

    def _shutdown_pool(self):
        """Shuts down the pool"""
        pass

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
        return f(*args, **kwargs)

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
        return True
        
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
        return future

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

    def _get_start_calc_message(
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
