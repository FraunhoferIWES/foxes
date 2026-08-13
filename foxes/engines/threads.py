from concurrent.futures import ThreadPoolExecutor

from .process import ProcessEngine, ProcessEngineRunner


class ThreadsEngineRunner(ProcessEngineRunner):
    """
    Engine runner for ThreadsEngine.

    :group: engines

    """

    def run(
        self,
        algo,
        model,
        mdata,
        *data,
        shared,
        chunk_store,
        chunk_key,
        out_dims,
        write_nc,
        write_chunk_ani=None,
        utm_zone=None,
        **cpars,
    ):
        """Helper function for running in a single process"""
        mdata = self._recombine_mdata_with_shared(mdata, shared)
        results = model.calculate(algo, mdata, *data, **cpars)
        cstore = (
            {chunk_key: algo.chunk_store[chunk_key]}
            if chunk_key in algo.chunk_store
            else {}
        )
        self._write_ani(algo, chunk_key, write_chunk_ani, mdata, *data)
        results = self._write_chunk_results(algo, results, write_nc, out_dims, mdata)

        return results, cstore


class ThreadsEngine(ProcessEngine):
    """
    The threads engine for foxes calculations.

    :group: engines

    """

    def __init__(self, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, share_cstore=True, supports_shared_data=False, **kwargs)

    def _create_pool(self):
        """Creates the pool"""
        self._pool = ThreadPoolExecutor(max_workers=self.n_workers, **self.pool_args)

    def new_runner(self):
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return ThreadsEngineRunner()
