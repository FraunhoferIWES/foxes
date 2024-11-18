import numpy as np
from xarray import Dataset

from foxes.algorithms import Iterative
from foxes.config import config
from foxes.core import get_engine
import foxes.variables as FV
import foxes.constants as FC

from . import models as mdls


class Sequential(Iterative):
    """
    A sequential calculation of states without chunking.

    This is of use for the evaluation in simulation
    environments that do not support multi-state computations,
    like FMUs.

    Attributes
    ----------
    ambient: bool
        Flag for ambient calculation
    calc_pars: dict
        Parameters for model calculation.
        Key: model name str, value: parameter dict
    states0: foxes.core.States
        The original states
    points: numpy.ndarray
        The points of interest, shape: (n_states, n_points, 3)
    plugins: list of foxes.algorithm.sequential.SequentialIterPlugin
        The plugins, updated with every iteration
    outputs: list of str
        The output variables
    :group: algorithms.sequential

    """

    @classmethod
    def get_model(cls, name):
        """
        Get the algorithm specific model

        Parameters
        ----------
        name: str
            The model name

        Returns
        -------
        model: foxes.core.model
            The model

        """
        try:
            return getattr(mdls, name)
        except AttributeError:
            return super().get_model(name)

    def __init__(
        self,
        farm,
        states,
        *args,
        points=None,
        ambient=False,
        calc_pars={},
        plugins=[],
        outputs=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.WindFarm
            The wind farm
        states: foxes.core.States
            The ambient states
        args: tuple, optional
            Additional arguments for Downwind
        points: numpy.ndarray, optional
            The points of interest, shape: (n_states, n_points, 3)
        ambient: bool
            Flag for ambient calculation
        calc_pars: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        plugins: list of foxes.algorithm.sequential.SequentialIterPlugin
            The plugins, updated with every iteration
        outputs: list of str, optional
            The output variables
        kwargs: dict, optional
            Additional arguments for Downwind

        """
        super().__init__(farm, mdls.SeqState(states), *args, **kwargs)
        self.ambient = ambient
        self.calc_pars = calc_pars
        self.states0 = self.states.states
        self.points = points
        self.plugins = plugins
        self.outputs = outputs if outputs is not None else self.DEFAULT_FARM_OUTPUTS

        self._verbo0 = self.verbosity + 1
        self.verbosity -= 1
        get_engine().verbosity -= 2

        self._i = None

    @property
    def iterating(self):
        """
        Flag for running iteration

        Returns
        -------
        itr: bool
            True if currently iterating

        """
        return self._i is not None

    def get_models_data(self, sel=None, isel=None):
        if sel is not None and len(sel):
            raise ValueError(f"calc_points does not support sel, got sel={sel}")
        if isel is not None and len(isel):
            raise ValueError(f"calc_points does not support isel, got isel={isel}")
        return self._model_data.isel({FC.STATE: [self.counter]})

    def __iter__(self):
        """Initialize the iterator"""

        if not self.iterating:
            if not self.initialized:
                self.initialize()
            self._print_deco("calc_farm")

            self._inds = self.states0.index()
            self._weights = self.states0.weights(self)
            self._i = 0
            self._counter = 0

            self._it = 0
            mlist, __ = self._collect_farm_models(
                None, self.calc_pars, ambient=self.ambient
            )
            self._calc_farm_vars(mlist)
            self._it = None

            self._model_data = Dataset(**super().get_models_idata())

            if self._verbo0 > 0:
                print("\nInput data:\n")
                print(self._model_data)
                print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
                print()

            self._farm_results = Dataset(
                coords={FC.STATE: self._model_data[FC.STATE].to_numpy()},
                data_vars={
                    v: (
                        (FC.STATE, FC.TURBINE),
                        np.zeros_like(self._model_data[FV.WEIGHT].to_numpy()),
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

    def __next__(self):
        """Run calculation for current step, then iterate to next"""

        if self._i < len(self._inds):

            self._counter = self._i
            self.states._counter = self._i
            self.states._size = 1
            self.states._indx = self._inds[self._i]
            self.states._weight = self._weights[self._i]

            if self._verbo0 > 0:
                print(f"{self.name}: Running state {self.states.index()[0]}")

            fres, fres_dnwnd = super().calc_farm(
                outputs=self.farm_vars,
                finalize=False,
                ret_dwnd_order=True,
                **self.calc_pars,
            )

            for v in self._farm_results.data_vars.keys():
                if FC.STATE in self._farm_results[v].dims:
                    self._farm_results[v].loc[{FC.STATE: [self.index]}] = fres[v]
                    self._farm_results_dwnd[v].loc[{FC.STATE: [self.index]}] = (
                        fres_dnwnd[v]
                    )

            if self.points is None:
                for p in self.plugins:
                    p.update(self, fres)

                self._i += 1
                return fres

            else:
                pres = super().calc_points(fres, points=self.points, finalize=False)

                if self._point_results is None:
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
                        self._point_results[v].loc[{FC.STATE: [self.index]}] = pres[v]

                for p in self.plugins:
                    p.update(self, fres, pres)

                self._i += 1
                return fres, pres

        else:
            del self._model_data

            self._i = None
            self.states._counter = None
            self.states._size = len(self._inds)
            self.states._indx = self._inds
            self.states._weight = self._weights

            for p in self.plugins:
                p.finalize(self)

            raise StopIteration

    @property
    def size(self):
        """
        The total number of iteration steps

        Returns
        -------
        s: int
            The total number of iteration steps

        """
        return self.states.size()

    @property
    def counter(self):
        """
        The current index counter

        Returns
        -------
        i: int
            The current index counter

        """
        return self._counter if self.iterating else None

    @property
    def index(self):
        """
        The current index

        Returns
        -------
        indx: int
            The current index

        """
        return self.states._indx if self.iterating else None

    def states_i0(self, counter, algo=None):
        """
        Returns counter or index

        Parameters
        ----------
        counter: bool
            Flag for counter
        algo: object, optional
            Dummy argument, due to consistency with
            foxes.core.Data.states_i0

        Returns
        -------
        i0: int
            The counter or index

        """
        return self.counter if counter else self.index

    @property
    def weight(self):
        """
        The current weight array

        Returns
        -------
        w: numpy.ndarray
            The current weight array, shape: (n_turbines,)

        """
        return self.states._weight if self.iterating else None

    @property
    def farm_results(self):
        """
        The overall farm results

        Returns
        -------
        results: xarray.Dataset
            The overall farm results

        """
        return self._farm_results

    @property
    def farm_results_downwind(self):
        """
        The overall farm results, with turbine
        dimension in downwind order

        Returns
        -------
        results: xarray.Dataset
            The overall farm results

        """
        return self._farm_results_dwnd

    @property
    def cur_farm_results(self):
        """
        The current farm results

        Returns
        -------
        results: xarray.Dataset
            The current farm results

        """
        return self._farm_results.isel({FC.STATE: [self.counter]})

    @property
    def point_results(self):
        """
        The overall point results

        Returns
        -------
        results: xarray.Dataset
            The overall point results

        """
        return self._point_results

    @property
    def cur_point_results(self):
        """
        The current point results

        Returns
        -------
        results: xarray.Dataset
            The current point results

        """
        return self._point_results.isel({FC.STATE: [self.counter]})

    def calc_farm(self):
        """
        Calculate farm data.

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        if not self.iterating:
            raise ValueError(f"calc_farm call is only allowed during iterations")
        return self.cur_farm_results

    def calc_points(
        self,
        farm_results,
        points,
        **kwargs,
    ):
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)
        points: numpy.ndarray
            The points of interest, shape: (n_states, n_points, 3)
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices

        Returns
        -------
        point_results: xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if not self.iterating:
            raise ValueError(f"calc_points call is only allowed during iterations")

        return super().calc_points(farm_results, points, finalize=False, **kwargs)
