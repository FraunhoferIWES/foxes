from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import Engine
import foxes.constants as FC


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
            verbosity=self.verbosity,
        )
        e.__enter__()
        return e, True

    def _select_engine_name(self, algo: Any = None, point_data: Any = None) -> str:
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

    def new_runner(self) -> Any:
        """
        Creates a new EngineRunner for running calculations in this engine.

        DefaultEngine delegates calculations to process or single engines,
        therefore it reuses the process-engine runner implementation.

        Returns
        -------
        runner
            The engine runner

        """
        e, temporary = self._get_delegate_process_engine()
        try:
            return e.new_runner()
        finally:
            self._release_delegate_process_engine(e, temporary)

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

    def run_calculation(
        self,
        algo: Any,
        model: Any,
        model_data: Any = None,
        farm_data: Any = None,
        point_data: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the model calculation

        Parameters
        ----------
        algo
            The algorithm object
        model
            The model that whose calculate function
            should be run
        model_data
            The initial model data
        farm_data
            The initial farm data
        point_data
            The initial point data

        Returns
        -------
        results
            The model results

        """
        if model_data is None:
            raise ValueError(f"Engine '{self.name}': Missing model_data")

        ename = self._select_engine_name(algo=algo, point_data=point_data)

        self.print(f"{type(self).__name__}: Selecting engine '{ename}'", level=1)

        # Reuse the delegated process engine directly to avoid nested engine
        # context entry while DefaultEngine itself is active.
        if ename == "process":
            e, temporary = self._get_delegate_process_engine()
            try:
                return e.run_calculation(
                    algo,
                    model,
                    model_data,
                    farm_data,
                    point_data=point_data,
                    **kwargs,
                )
            finally:
                self._release_delegate_process_engine(e, temporary)

        suspended_delegate = (
            hasattr(self, "_delegate_process_engine")
            and self._delegate_process_engine is not None
        )
        delegate = self._delegate_process_engine if suspended_delegate else None
        if suspended_delegate:
            assert delegate is not None
            delegate.__exit__(None, None, None)

        try:
            with Engine.new(
                ename,
                n_procs=self.n_procs,
                chunk_size_states=self.chunk_size_states,
                chunk_size_points=self.chunk_size_points,
                verbosity=self.verbosity,
            ) as e:
                results = e.run_calculation(
                    algo, model, model_data, farm_data, point_data=point_data, **kwargs
                )
        finally:
            if suspended_delegate:
                assert delegate is not None
                delegate.__enter__()

        return results
