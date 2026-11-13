from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
from foxes.core import Engine
from foxes.core.engine import _Calculation
from foxes.utils import Dict

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, Output
    from xarray import Dataset


@dataclass
class _DefaultCalculation(_Calculation):
    """Delegate and context ownership for one default-engine calculation."""

    resources: ExitStack = field(default_factory=ExitStack)
    backend: Engine | None = None
    backend_calculation: _Calculation | None = None


class DefaultEngine(Engine):
    """
    The case size dependent default engine.
    """

    _delegate_process_engine: Engine | None
    _entered: bool

    def __enter__(self) -> DefaultEngine:
        self._delegate_process_engine = Engine.new(
            "process",
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            progress_bar=self.progress_bar,
            verbosity=self.verbosity,
        )
        self._delegate_process_engine.__enter__()
        self._entered = True
        return self

    def __exit__(self, *exit_args: Any) -> None:
        if not hasattr(self, "_entered") or not self._entered:
            raise ValueError(
                f"Engine '{self.name}': Exit called for not entered engine"
            )
        delegate = getattr(self, "_delegate_process_engine", None)
        if delegate is not None:
            delegate.__exit__(*exit_args)
            self._delegate_process_engine = None
        self._entered = False

    def _get_delegate_process_engine(self) -> tuple[Engine, bool]:
        """Returns the delegated process engine, creating a temporary one if needed."""
        if (
            hasattr(self, "_delegate_process_engine")
            and self._delegate_process_engine is not None
        ):
            return self._delegate_process_engine, False
        e = Engine.new(
            "process",
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            progress_bar=self.progress_bar,
            verbosity=self.verbosity,
        )
        e.__enter__()
        return e, True

    def _select_engine_name(
        self, algo: Algorithm | None = None, point_data: Any = None
    ) -> str:
        """Selects SingleChunkEngine vs ProcessEngine where possible."""
        if algo is None:
            return "process"

        max_n = np.sqrt(self.n_workers) * (500 / algo.n_turbines) ** 1.5
        if (algo.n_states >= max_n) or (
            point_data is not None
            and algo.n_states * point_data.sizes[FC.TARGET] > 10000
        ):
            return "process"

        return "single"

    def _release_delegate_process_engine(self, engine: Engine, temporary: bool) -> None:
        """Releases temporary delegated process engine instances."""
        if temporary:
            engine.__exit__(None, None, None)

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
        e, temporary = self._get_delegate_process_engine()
        try:
            return e.submit(f, *args, **kwargs)
        finally:
            self._release_delegate_process_engine(e, temporary)

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
        e, temporary = self._get_delegate_process_engine()
        try:
            return e.future_is_done(future)
        finally:
            self._release_delegate_process_engine(e, temporary)

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
        e, temporary = self._get_delegate_process_engine()
        try:
            return e.await_result(future)
        finally:
            self._release_delegate_process_engine(e, temporary)

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
        e, temporary = self._get_delegate_process_engine()
        try:
            return e.map(func, inputs, *args, **kwargs)
        finally:
            self._release_delegate_process_engine(e, temporary)

    def _create_calculation(self, chunk_store: dict[Any, Any]) -> _DefaultCalculation:
        return _DefaultCalculation(chunk_store=chunk_store)

    def _cleanup_calculation(self, calculation: _Calculation, failed: bool) -> None:
        # Finish delegated workers before leaving or restoring engine contexts.
        assert isinstance(calculation, _DefaultCalculation)
        try:
            if calculation.backend is not None:
                assert calculation.backend_calculation is not None
                calculation.backend._cleanup_calculation(
                    calculation.backend_calculation, failed
                )
        finally:
            calculation.resources.close()

    def _run_calculation(
        self,
        calculation: _Calculation,
        algo: Algorithm,
        model: DataCalcModel,
        results: Dict[str, Any],
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        *,
        extra_data: dict[str, Any] | None = None,
        output: Output | None = None,
        sel: dict[str, Any] | None = None,
        iterative: bool = False,
        progress_label: str | None = None,
        **calc_pars: Any,
    ) -> Dict[str, Any]:
        """Delegate backend hooks without nesting the common output lifecycle."""
        assert isinstance(calculation, _DefaultCalculation)
        if model_data is None:
            raise ValueError(f"Engine '{self.name}': Missing model_data")

        ename = self._select_engine_name(algo=algo, point_data=point_data)

        self.print(f"{type(self).__name__}: Selecting engine '{ename}'", level=1)

        # Reuse the delegated process engine directly to avoid nested engine
        # context entry while DefaultEngine itself is active.
        if ename == "process":
            e, temporary = self._get_delegate_process_engine()
            calculation.resources.callback(
                self._release_delegate_process_engine, e, temporary
            )
        else:
            delegate = getattr(self, "_delegate_process_engine", None)
            if delegate is not None:
                delegate.__exit__(None, None, None)
                calculation.resources.callback(delegate.__enter__)
            e = calculation.resources.enter_context(
                Engine.new(
                    ename,
                    n_procs=self.n_procs,
                    chunk_size_states=self.chunk_size_states,
                    chunk_size_points=self.chunk_size_points,
                    progress_bar=self.progress_bar,
                    verbosity=self.verbosity,
                )
            )

        # Share cache updates and chunk counts with the common finalization.
        backend_calculation = e._create_calculation(calculation.chunk_store)
        calculation.backend_calculation = backend_calculation
        calculation.backend = e
        calculation.new_chunk_store = backend_calculation.new_chunk_store
        results = e._run_calculation(
            backend_calculation,
            algo,
            model,
            results,
            model_data,
            farm_data,
            point_data,
            extra_data=extra_data,
            output=output,
            sel=sel,
            iterative=iterative,
            progress_label=progress_label,
            **calc_pars,
        )
        calculation.n_chunks_states = backend_calculation.n_chunks_states
        calculation.n_chunks_targets = backend_calculation.n_chunks_targets
        return results
