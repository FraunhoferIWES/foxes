from foxes.utils import import_module

from .pool import PoolEngine

Pool = None


def load_multiprocess():
    """On-demand loading of the multiprocess package"""
    global Pool
    if Pool is None:
        Pool = import_module("multiprocess", hint="pip install multiprocess").Pool


class MultiprocessEngine(PoolEngine):
    """
    The multiprocessing engine for foxes calculations.

    :group: engines

    """

    def _create_pool(self):
        """Creates the pool"""
        load_multiprocess()
        self._pool = Pool(processes=self.n_procs)

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
        return self._pool.apply_async(f, args=args, kwds=kwargs)

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
        return future.get()

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self._pool.close()
        self._pool.terminate()
        self._pool.join()
