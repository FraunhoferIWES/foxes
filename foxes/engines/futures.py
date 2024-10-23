from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .pool import PoolEngine


class ThreadsEngine(PoolEngine):
    """
    The threads engine for foxes calculations.

    :group: engines

    """

    def _create_pool(self):
        """Creates the pool"""
        self._pool = ThreadPoolExecutor(max_workers=self.n_procs)

    def _submit(self, f, *args, **kwargs):
        """
        Submits to the pool

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
        return self._pool.submit(f, *args, **kwargs)

    def _result(self, future):
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
        return future.result()

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self._pool.shutdown()


class ProcessEngine(ThreadsEngine):
    """
    The processes engine for foxes calculations.

    :group: engines

    """

    def _create_pool(self):
        """Creates the pool"""
        self._pool = ProcessPoolExecutor(max_workers=self.n_procs)
