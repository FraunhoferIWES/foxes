from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from xarray import Dataset
from typing import TYPE_CHECKING, Any, cast

from foxes.algorithms import Iterative
from foxes.config import config
from foxes.core import get_engine
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

    Attributes
    ----------
    ambient
        Flag for ambient calculation
    calc_pars
        Parameters for model calculation.
        Key: model name str, value: parameter dict
    states0
        The original states
    points
        The points of interest, shape: (n_states, n_points, 3)
    plugins
        The plugins, updated with every iteration
    outputs
        The output variables
    :group: algorithms.sequential

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
        Constructor.

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
        self.outputs = outputs if outputs is not None else self.DEFAULT_FARM_OUTPUTS

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
        assert self.counter is not None
        return self._model_data.isel({FC.STATE: [self.counter]}), self.loaded_data[
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
            mlist, __ = self._collect_farm_models(
                None, self.calc_pars, ambient=self.ambient
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
        """Run calculation for current step, then iterate to next"""

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
                    outputs=self.farm_vars,
                    finalize=False,
                    ret_dwnd_order=True,
                    **self.calc_pars,
                ),
            )
            assert fres_dnwnd is not None

            assert self._farm_results is not None
            assert self._farm_results_dwnd is not None
            for v in self._farm_results.data_vars.keys():
                if FC.STATE in self._farm_results[v].dims:
                    self._farm_results[v].loc[{FC.STATE: [sstate._indx]}] = fres[v]
                    self._farm_results_dwnd[v].loc[{FC.STATE: [sstate._indx]}] = (
                        fres_dnwnd[v]
                    )

            if self.points is None:
                for p in self.plugins:
                    p.update(self, fres)

                assert self._i is not None
                self._i += 1
                return fres

            else:
                pres = cast(
                    Dataset,
                    super().calc_points(fres, points=self.points, finalize=False),
                )

                if self._point_results is None:
                    assert self._model_data is not None
                    n_states = self._model_data.sizes[FC.STATE]
                    self._point_results = Dataset(
                        coords={
                            FC.STATE: self._model_data[FC.STATE].to_numpy(),
                            **{c: d for c, d in pres.coords.items() if c != FC.STATE},
                        },
                        data_vars={
                            v: (
                                d.dims,
                                np.zeros([n_states] + list(d.shape[1:]), dtype=d.dtype),
                            )
                            for v, d in pres.data_vars.items()
                            if d.dims[0] == FC.STATE
                        },
                    )
                    for v, d in pres.data_vars.items():
                        if FC.STATE not in d.dims:
                            self._point_results[v] = d

                for v in self._point_results.data_vars.keys():
                    if FC.STATE in self._point_results[v].dims:
                        assert self.counter is not None
                        self._point_results[v].loc[{FC.STATE: [self.counter]}] = pres[v]

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

    def states_i0(self, counter: bool, algo: Any = None) -> Any:
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
        assert self.counter is not None
        return self._farm_results.isel({FC.STATE: [self.counter]})

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
        assert self.counter is not None
        return self._point_results.isel({FC.STATE: [self.counter]})

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
        points: np.ndarray,
        **kwargs: Any,
    ) -> Dataset:
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results
            The farm results. The calculated variables have
            dimensions (state, turbine)
        points
            The points of interest, shape: (n_states, n_points, 3)
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices

        Returns
        -------
        point_results
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if not self.iterating:
            raise ValueError("calc_points call is only allowed during iterations")

        return super().calc_points(farm_results, points, finalize=False, **kwargs)
