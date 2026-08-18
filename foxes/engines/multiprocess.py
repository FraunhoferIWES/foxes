from __future__ import annotations

from typing import Any

from foxes.utils import import_module

from .process import ProcessEngineRunner
from .pool import PoolEngine


class MultiprocessEngine(PoolEngine):
    """
    The multiprocessing engine for foxes calculations.


    """

    def new_runner(self) -> ProcessEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner
            The engine runner

        """
        return ProcessEngineRunner()

    def _create_pool(self) -> None:
        """Creates the pool"""
        Pool = import_module("multiprocess").Pool
        self._pool = Pool(processes=self.n_workers, **self.pool_args)

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> Any:
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
        return self._pool.apply_async(f, args=args, kwds=kwargs)

    def future_is_done(self, future: Any) -> bool:
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
        return future.ready()

    def await_result(self, future: Any) -> Any:
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
        return future.get()

    def _shutdown_pool(self) -> None:
        """Shuts down the pool"""
        self._pool.close()
        self._pool.terminate()
        self._pool.join()
