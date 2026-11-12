from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from xarray import Dataset
from typing import TYPE_CHECKING, Any, cast

from foxes.algorithms import Iterative
from foxes.config import config
from foxes.core import Algorithm, get_engine
from foxes.models.outputs import FarmResultsOutput, PointResultsOutput
import foxes.variables as FV
import foxes.constants as FC

from . import models as mdls

if TYPE_CHECKING:
    from foxes.core import States, WindFarm
    from .models.plugin import SequentialPlugin


class Sequential(Iterative):
    """
    A sequential calculation of states without chunking.

    This is of use for the evaluation in simulation
    environments that do not support multi-state computations,
    like FMUs.
    """

    @classmethod
    def get_model(cls, name: str) -> Any:
        """
        Get the algorithm specific model

        Parameters
        ----------
        name
            The model name

        Returns
        -------
        model
            The model

        """
        try:
            return getattr(mdls, name)
        except AttributeError:
            return super().get_model(name)

    def __init__(
        self,
        farm: WindFarm,
        states: States,
        *args: Any,
        points: np.ndarray | None = None,
        ambient: bool = False,
        calc_pars: dict[str, Any] = {},
        plugins: list[SequentialPlugin] = [],
        outputs: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        farm
            The wind farm
        states
            The ambient states
        args
            Additional arguments for Downwind
        points
            The points of interest, shape: (n_states, n_points, 3)
        ambient
            Flag for ambient calculation
        calc_pars
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        plugins
            The plugins, updated with every iteration
        outputs
            The output variables
        kwargs
            Additional arguments for Downwind
        """
        super().__init__(farm, mdls.SeqState(states), *args, **kwargs)
        self.ambient = ambient
        self.calc_pars = calc_pars
        self.states0 = self._seq_state().states
        self.points = points
        self.plugins = plugins
        self.outputs = outputs

        self._verbo0 = self.verbosity + 1
        self.verbosity -= 1

        self._i: int | None = None
        self._counter: int | None = None
        self._inds: Any = None
        self._farm_results: Dataset | None = None
        self._farm_results_dwnd: Dataset | None = None
        self._point_results: Dataset | None = None
        self._model_data: Dataset | None = None

    def _seq_state(self) -> mdls.SeqState:
        sstate = self.states
        assert isinstance(sstate, mdls.SeqState)
        return sstate

    @property
    def iterating(self) -> bool:
        """
        Flag for running iteration

        Returns
        -------
        iterating
            True if currently iterating

        """
        return self._i is not None

    def get_model_data(self, pop: bool = False) -> tuple[Dataset, dict[str, Any]]:
        if self._model_data is None:
            return super().get_model_data(pop=pop)
        assert self.index is not None
        return self._model_data.sel({FC.STATE: [self.index]}), self.loaded_data[
            "extra_data"
        ]

    def __iter__(self) -> Sequential:
        """Initialize the iterator"""

        if not self.iterating:
            # Adjust verbosity if engine is set
            try:
                eng = get_engine()
                if eng is not None:
                    eng.verbosity -= 2
            except ValueError:
                pass

            if not self.initialized:
                self.initialize()
            self.print_deco("calc_farm")

            self._inds = self.states0.index()
            self._i = 0
            self._counter = 0

            self._it = cast(Any, 0)
            self._final_run = False
            mlist, __ = self._collect_farm_models(
                self.calc_pars, ambient=self.ambient, output=FarmResultsOutput()
            )
            self._calc_farm_vars(mlist)
            self._it = cast(Any, None)

            self._model_data, _ = super().get_model_data(pop=False)

            if self._verbo0 > 0:
                print("\nInput data:\n")
                print(self._model_data)
                print("\nOutput farm variables:", ", ".join(self.farm_vars))
                print()

            self._farm_results = Dataset(
                coords={FC.STATE: self._inds},
                data_vars={
                    v: (
                        (FC.STATE, FC.TURBINE),
                        np.zeros(
                            (len(self._inds), self.n_turbines),
                            dtype=config.dtype_double,
                        ),
                    )
                    for v in self.farm_vars
                },
            )
            self._farm_results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
            if FV.ORDER in self._farm_results:
                self._farm_results[FV.ORDER] = self._farm_results[FV.ORDER].astype(
                    config.dtype_int
                )
            self._farm_results_dwnd = self._farm_results.copy(deep=True)

            self._point_results = None

            for p in self.plugins:
                p.initialize(self)

        return self

    def __next__(self) -> Dataset | tuple[Dataset, Dataset]:
        """Calculate the current state and advance the sequential iterator.

        Returns
        -------
        state_results
            Current farm results on ``(FC.STATE, FC.TURBINE)`` in original
            turbine order, or a pair of farm and point datasets when iterator
            points are configured. Point fields use ``(FC.STATE, FC.POINT)``.
            Each dataset contains one state; point order is unchanged.

        Raises
        ------
        StopIteration
            After all states finish and plugins are finalized.

        Notes
        -----
        Accumulated farm fields, point fields and state-dependent point
        coordinates are updated in iterator-owned result datasets. Plugins
        receive each state's results while its wake history is available.
        """

        assert self._i is not None
        assert self._inds is not None
        if self._i < len(self._inds):
            sstate = self._seq_state()
            self._counter = self._i
            sstate._counter = self._i
            sstate._size = 1
            sstate._indx = self._inds[self._i]
            assert sstate._indx is not None

            if self._verbo0 > 0:
                print(f"{self.name}: Running state {self.states.index()[0]}")

            self.reset_chunk_store()
            fres, fres_dnwnd = cast(
                tuple[Dataset, Dataset],
                super().calc_farm(
                    outputs=FarmResultsOutput(variables=self.farm_vars),
                    finalize=False,
                    ret_dwnd_order=True,
                    **self.calc_pars,
                ),
            )
            assert fres_dnwnd is not None

            assert self._farm_results is not None
            assert self._farm_results_dwnd is not None
            for v in self._farm_results.data_vars.keys():
                self._farm_results[v].attrs.update(fres[v].attrs)
                if FC.STATE in self._farm_results[v].dims:
                    self._farm_results[v].loc[{FC.STATE: [sstate._indx]}] = fres[v]
                    self._farm_results_dwnd[v].loc[{FC.STATE: [sstate._indx]}] = (
                        fres_dnwnd[v]
                    )
            self._farm_results.attrs.update(fres.attrs)

            if self.points is None:
                for p in self.plugins:
                    p.update(self, fres)

                assert self._i is not None
                self._i += 1
                return fres

            else:
                pres = cast(
                    Dataset,
                    super().calc_points(
                        fres,
                        points=self.points,
                        outputs=PointResultsOutput(),
                        finalize=False,
                    ),
                )

                if self._point_results is None:
                    self._point_results = self._new_point_results(pres)

                # Retain moving point locations alongside each state's fields.
                for c, d in pres.coords.items():
                    if c != FC.STATE and FC.STATE in d.dims:
                        self._point_results[c].loc[{FC.STATE: [sstate._indx]}] = d

                for v in self._point_results.data_vars.keys():
                    if FC.STATE in self._point_results[v].dims:
                        assert self.index is not None
                        self._point_results[v].loc[{FC.STATE: [self.index]}] = pres[v]

                for p in self.plugins:
                    p.update(self, fres, pres)

                assert self._i is not None
                self._i += 1
                return fres, pres

        else:
            del self._model_data
            sstate = self._seq_state()

            self._i = None
            sstate._counter = None
            sstate._size = len(self._inds)
            sstate._indx = self._inds

            for p in self.plugins:
                p.finalize(self)

            # Reset verbosity if engine is set
            try:
                eng = get_engine()
                if eng is not None:
                    eng.verbosity += 2
            except ValueError:
                pass

            raise StopIteration

    def _new_point_results(self, point_results: Dataset) -> Dataset:
        """Allocate iterator-owned point buffers from the first state's results.

        Parameters
        ----------
        point_results
            First-step dataset whose fields lead with ``FC.STATE`` when they
            vary by state. Coordinates may place ``FC.STATE`` on any axis.
            The input dataset is not mutated.

        Returns
        -------
        accumulated_results
            Dataset with the full model-state coordinate. State-dependent fields
            and coordinates receive zero-filled arrays of the original dtype;
            static values and supplied point order are retained. Static arrays
            may share storage with ``point_results``.
        """
        # Expand only state-dependent arrays to the iterator's full state axis.
        assert self._model_data is not None
        n_states = self._model_data.sizes[FC.STATE]
        results = Dataset(
            coords={
                FC.STATE: self._model_data[FC.STATE].to_numpy(),
                **{
                    name: (
                        values.dims,
                        np.zeros(
                            [
                                n_states if dim == FC.STATE else size
                                for dim, size in zip(values.dims, values.shape)
                            ],
                            dtype=values.dtype,
                        ),
                    )
                    if FC.STATE in values.dims
                    else values
                    for name, values in point_results.coords.items()
                    if name != FC.STATE
                },
            },
            data_vars={
                name: (
                    values.dims,
                    np.zeros([n_states] + list(values.shape[1:]), dtype=values.dtype),
                )
                for name, values in point_results.data_vars.items()
                if values.dims[0] == FC.STATE
            },
        )

        # Preserve state-independent variables without allocating repeated data.
        for name, values in point_results.data_vars.items():
            if FC.STATE not in values.dims:
                results[name] = values

        # Array-based allocation does not carry xarray variable attributes.
        for name in results.variables:
            results[name].attrs.update(point_results[name].attrs)
        results.attrs.update(point_results.attrs)

        return results

    @property
    def size(self) -> int:
        """
        The total number of iteration steps

        Returns
        -------
        size
            The total number of iteration steps

        """
        return self.states.size()

    @property
    def counter(self) -> int | None:
        """
        The current index counter

        Returns
        -------
        counter
            The current index counter

        """
        return self._counter if self.iterating else None

    @property
    def index(self) -> Any:
        """
        The current index

        Returns
        -------
        index
            The current index

        """
        return self._seq_state()._indx if self.iterating else None

    def states_i0(self, counter: bool, algo: Algorithm | None = None) -> Any:
        """
        Returns counter or index

        Parameters
        ----------
        counter
            Flag for counter
        algo
            Dummy argument, due to consistency with
            foxes.core.Data.states_i0

        Returns
        -------
        i0
            The counter or index

        """
        return self.counter if counter else self.index

    @property
    def farm_results(self) -> Dataset:
        """
        The overall farm results

        Returns
        -------
        results
            The overall farm results

        """
        assert self._farm_results is not None
        return self._farm_results

    @property
    def farm_results_downwind(self) -> Dataset:
        """
        The overall farm results, with turbine
        dimension in downwind order

        Returns
        -------
        results
            The overall farm results

        """
        assert self._farm_results_dwnd is not None
        return self._farm_results_dwnd

    @property
    def cur_farm_results(self) -> Dataset:
        """
        The current farm results

        Returns
        -------
        results
            The current farm results

        """
        assert self._farm_results is not None
        assert self.index is not None
        return self._farm_results.sel({FC.STATE: [self.index]})

    @property
    def point_results(self) -> Dataset | None:
        """
        The overall point results

        Returns
        -------
        results
            The overall point results

        """
        return self._point_results

    @property
    def cur_point_results(self) -> Dataset:
        """
        The current point results

        Returns
        -------
        results
            The current point results

        """
        assert self._point_results is not None
        assert self.index is not None
        return self._point_results.sel({FC.STATE: [self.index]})

    def calc_farm(self) -> Dataset:
        """
        Calculate farm data.

        Returns
        -------
        farm_results
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        if not self.iterating:
            raise ValueError("calc_farm call is only allowed during iterations")
        return self.cur_farm_results

    def calc_points(
        self,
        farm_results: Dataset,
        points: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Calculate selected point outputs during the current iterator step.

        Parameters
        ----------
        farm_results
            Current-state farm fields on ``(FC.STATE, FC.TURBINE)`` in original
            turbine order.
        points
            Static ``(n_points, 3)`` or state-dependent
            ``(n_states, n_points, 3)`` locations in metres, with components
            ``x``, ``y`` and height. May be ``None`` when ``point_results`` is
            supplied through ``kwargs``.
        kwargs
            ``Downwind.calc_points`` options, including model-book output names,
            output instances or lists, and precomputed point results.
            ``finalize`` is always forced to false so the iterator remains usable.

        Returns
        -------
        results
            Selected output value or result-name mapping, following
            ``Downwind.calc_points``. The default dataset uses
            ``(FC.STATE, FC.POINT)`` and preserves supplied point order.

        Raises
        ------
        ValueError
            If called outside an active iteration.

        """
        if not self.iterating:
            raise ValueError("calc_points call is only allowed during iterations")

        kwargs["finalize"] = False
        return super().calc_points(farm_results, points, **kwargs)
