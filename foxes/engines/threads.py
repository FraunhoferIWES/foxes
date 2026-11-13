from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .process import ProcessEngine, ProcessEngineRunner

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, FData, MData, TData, Output


class ThreadsEngineRunner(ProcessEngineRunner):
    """
    Engine runner for ThreadsEngine.
    """

    def run(
        self,
        algo: Algorithm,
        model: DataCalcModel | None,
        mdata: MData,
        fdata: FData,
        tdata: TData | None = None,
        output: Output | None = None,
        *,
        shared: Any,
        chunk_store: dict[Any, Any],
        chunk_key: Any,
        out_dims: tuple[str, ...],
        utm_zone: tuple[int, str] | None = None,
        **cpars: Any,
    ) -> tuple[dict[str, Any] | None, dict[Any, Any]]:
        """Calculate one thread chunk and evaluate its output model.

        Parameters
        ----------
        algo
            Running algorithm with an engine-managed shared chunk store.
        model
            Initialized scientific model, or ``None`` for output-only chunks.
        mdata
            Chunk model data, recombined with shared arrays before calculation.
        fdata
            Chunk farm data, replaced by previous farm results when available.
        tdata
            Chunk targets, or ``None`` for a farm calculation.
        output
            Output model consuming prepared data and scientific results.
            ``None`` skips output evaluation.
        shared
            Shared-storage manifest, or ``None`` for entirely local model data.
        chunk_store
            Engine-managed scientific caches, already attached to ``algo``.
        chunk_key
            State/target chunk index identifying returned cache entries.
        out_dims
            Scientific model's output dimensions.
        utm_zone
            Engine-supplied zone, unused because threads share process configuration.
        cpars
            Parameters forwarded to the scientific model's ``calculate`` method.

        Returns
        -------
        results
            Output payload mapping, or ``None`` when no output is selected.
        chunk_store
            This chunk's caches, including entries created by the output.
        """
        mdata = self._recombine_mdata_with_shared(mdata, shared)

        fdata, _ = self._apply_prev_farm_results(algo, mdata, fdata)

        model_results: dict[str, Any] | None
        if model is None:
            model_results = None
        elif tdata is None:
            model_results = model.calculate(algo, mdata, fdata, **cpars)
        else:
            model_results = model.calculate(algo, mdata, fdata, tdata, **cpars)

        if output is None:
            results = None
        else:
            results = output.calculate(
                algo=algo,
                mdata=mdata,
                fdata=fdata,
                tdata=tdata,
                model_results=model_results,
            )

        # Return caches created by both scientific models and chunk outputs.
        cstore = (
            {chunk_key: algo.chunk_store[chunk_key]}
            if chunk_key in algo.chunk_store
            else {}
        )
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
