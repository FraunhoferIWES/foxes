from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .process import ProcessEngine, ProcessEngineRunner

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, FData, MData, TData


class ThreadsEngineRunner(ProcessEngineRunner):
    """
    Engine runner for ThreadsEngine.
    """

    def run(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        mdata: MData,
        fdata: FData,
        tdata: TData | None = None,
        *,
        shared: Any,
        chunk_store: dict[Any, Any],
        chunk_key: Any,
        out_dims: tuple[str, ...],
        write_nc: dict[str, Any] | None,
        write_chunk_ani: dict[str, Any] | None = None,
        utm_zone: tuple[int, str] | None = None,
        **cpars: Any,
    ) -> tuple[dict[str, Any] | None, dict[Any, Any]]:
        """Helper function for running in a single process"""
        mdata = self._recombine_mdata_with_shared(mdata, shared)

        fdata, has_prev_farm_results = self._apply_prev_farm_results(algo, mdata, fdata)

        results: dict[str, Any] | None
        if tdata is None:
            results = model.calculate(algo, mdata, fdata, **cpars)
        else:
            results = model.calculate(algo, mdata, fdata, tdata, **cpars)
        results = self._merge_prev_farm_results(has_prev_farm_results, fdata, results)

        cstore = (
            {chunk_key: algo.chunk_store[chunk_key]}
            if chunk_key in algo.chunk_store
            else {}
        )
        self._write_ani(algo, chunk_key, write_chunk_ani, mdata, fdata, tdata)
        results = self._write_chunk_results(algo, results, write_nc, out_dims, mdata)

        return results, cstore


class ThreadsEngine(ProcessEngine):
    """
    The threads engine for foxes calculations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor
        """
        super().__init__(*args, share_cstore=True, supports_shared_data=False, **kwargs)

    def _create_pool(self) -> None:
        """Creates the pool"""
        self._pool = ThreadPoolExecutor(max_workers=self.n_workers, **self.pool_args)

    def new_runner(self) -> ThreadsEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner
            The engine runner

        """
        return ThreadsEngineRunner()
