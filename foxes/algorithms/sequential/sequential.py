import numpy as np
from xarray import Dataset

from foxes.algorithms.downwind.downwind import Downwind
import foxes.constants as FC
import foxes.variables as FV
from foxes.core.data import MData, FData, TData

from . import models as mdls


class Sequential(Downwind):
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
        chunks={FC.STATE: None, FC.POINT: 4000},
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
        chunks: dict
            The xarray.Dataset chunk parameters
        plugins: list of foxes.algorithm.sequential.SequentialIterPlugin
            The plugins, updated with every iteration
        outputs: list of str, optional
            The output variables
        kwargs: dict, optional
            Additional arguments for Downwind

        """
        super().__init__(farm, mdls.SeqState(states), *args, chunks=chunks, **kwargs)
        self.ambient = ambient
        self.calc_pars = calc_pars
        self.states0 = self.states.states
        self.points = points
        self.plugins = plugins
        self.outputs = outputs if outputs is not None else self.DEFAULT_FARM_OUTPUTS

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

            self._mlist, self._calc_pars = self._collect_farm_models(
                self.outputs, self.calc_pars, self.ambient
            )
            if not self._mlist.initialized:
                self._mlist.initialize(self, self.verbosity)
                self._calc_farm_vars(self._mlist)
            self._print_model_oder(self._mlist, self._calc_pars)

            self._mdata = self.get_models_idata()
            if self.verbosity > 0:
                s = "\n".join(
                    [
                        f"  {v}: {d[0]} {d[1].dtype}, shape {d[1].shape}"
                        for v, d in self._mdata["data_vars"].items()
                    ]
                )
                print("\nInput data:\n")
                print(s, "\n")
                print(f"Output farm variables:", ", ".join(self.farm_vars))
                print()

            self._mdata = MData(
                data={v: d[1] for v, d in self._mdata["data_vars"].items()},
                dims={v: d[0] for v, d in self._mdata["data_vars"].items()},
                loop_dims=[FC.STATE],
                name="mdata",
            )

            self._fdata = FData(
                data={
                    v: np.zeros((self.n_states, self.n_turbines), dtype=FC.DTYPE)
                    for v in self.farm_vars
                },
                dims={v: (FC.STATE, FC.TURBINE) for v in self.farm_vars},
                loop_dims=[FC.STATE],
                name="fdata",
            )

            if self.points is not None:
                self._plist, self._calc_pars_p = self._collect_point_models(
                    ambient=self.ambient
                )
                if not self._plist.initialized:
                    self._plist.initialize(self, self.verbosity)
                self._pvars = self._plist.output_point_vars(self)
                self.print(f"\nOutput point variables:", ", ".join(self._pvars), "\n")

                n_points = self.points.shape[1]
                self._tdata = TData.from_points(
                    self.points,
                    data={
                        v: np.zeros((self.n_states, n_points, 1), dtype=FC.DTYPE)
                        for v in self._pvars
                    },
                    dims={v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in self._pvars},
                )

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

            mdata = MData(
                data={
                    v: d[self._i, None] if self._mdata.dims[v][0] == FC.STATE else d
                    for v, d in self._mdata.items()
                },
                dims={v: d for v, d in self._mdata.dims.items()},
                loop_dims=[FC.STATE],
                name="mdata",
            )

            fdata = FData(
                data={
                    v: np.zeros((1, self.n_turbines), dtype=FC.DTYPE)
                    for v in self.farm_vars
                },
                dims={v: (FC.STATE, FC.TURBINE) for v in self.farm_vars},
                loop_dims=[FC.STATE],
                name="fdata",
            )

            fres = self._mlist.calculate(self, mdata, fdata, parameters=self._calc_pars)
            fres[FV.WEIGHT] = self.weight[None, :]

            for v, d in fres.items():
                self._fdata[v][self._i] = d[0]

            fres = Dataset(
                coords={FC.STATE: [self.index]},
                data_vars={v: ((FC.STATE, FC.TURBINE), d) for v, d in fres.items()},
            )
            fres[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
            if FV.ORDER in fres:
                fres[FV.ORDER] = fres[FV.ORDER].astype(FC.ITYPE)

            if self.points is None:
                for p in self.plugins:
                    p.update(self, fres)

                self._i += 1
                return fres

            else:
                n_points = self.points.shape[1]
                tdata = TData.from_points(
                    self.points[self.counter, None],
                    data={
                        v: np.zeros((1, n_points, 1), dtype=FC.DTYPE)
                        for v in self._pvars
                    },
                    dims={v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in self._pvars},
                )

                pres = self._plist.calculate(
                    self, mdata, fdata, tdata, parameters=self._calc_pars_p
                )

                for v, d in pres.items():
                    self._tdata[v][self._i] = d[0]

                pres = Dataset(
                    coords={FC.STATE: [self.index]},
                    data_vars={
                        v: ((FC.STATE, FC.TARGET, FC.TPOINT), d)
                        for v, d in pres.items()
                    },
                )

                for p in self.plugins:
                    p.update(self, fres, pres)

                self._i += 1
                return fres, pres

        else:
            del self._mdata

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
    def mdata(self):
        """
        Get the current model data

        Returns
        -------
        d: foxes.core.MData
            The current model data

        """
        return self._mdata if self.iterating else None

    @property
    def fdata(self):
        """
        Get the current farm data

        Returns
        -------
        d: foxes.core.FData
            The current farm data

        """
        return self._fdata

    @property
    def tdata(self):
        """
        Get the current point data

        Returns
        -------
        d: foxes.core.TData
            The current point data

        """
        return self._tdata if self.points is not None and self.iterating else None

    @property
    def farm_results(self):
        """
        The overall farm results

        Returns
        -------
        results: xarray.Dataset
            The overall farm results

        """
        results = Dataset(
            coords={FC.STATE: self._inds, FC.TURBINE: np.arange(self.n_turbines)},
            data_vars={v: (self._fdata.dims[v], d) for v, d in self._fdata.items()},
        )

        results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
        if FV.ORDER in results:
            results[FV.ORDER] = results[FV.ORDER].astype(FC.ITYPE)

        return results

    @property
    def prev_farm_results(self):
        """
        Alias for farm_results

        Returns
        -------
        results: xarray.Dataset
            The overall farm results

        """
        return self.farm_results

    @property
    def cur_farm_results(self):
        """
        The current farm results

        Returns
        -------
        results: xarray.Dataset
            The current farm results

        """

        i = self.counter
        results = Dataset(
            coords={FC.STATE: [self.index], FC.TURBINE: np.arange(self.n_turbines)},
            data_vars={
                v: (self._fdata.dims[v], d[i, None]) for v, d in self._fdata.items()
            },
        )

        results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
        if FV.ORDER in results:
            results[FV.ORDER] = results[FV.ORDER].astype(FC.ITYPE)

        return results

    @property
    def point_results(self):
        """
        The overall point results

        Returns
        -------
        results: xarray.Dataset
            The overall point results

        """
        n_points = self.points.shape[1]
        results = Dataset(
            coords={
                FC.STATE: self._inds,
                FC.TURBINE: np.arange(self.n_turbines),
                FC.POINT: np.arange(n_points),
                FC.XYH: np.arange(3),
            },
            data_vars={v: (self._tdata.dims[v], d) for v, d in self._tdata.items()},
        )

        return results

    @property
    def cur_point_results(self):
        """
        The current point results

        Returns
        -------
        results: xarray.Dataset
            The current point results

        """

        n_points = self.points.shape[1]
        i = self.counter

        results = Dataset(
            coords={
                FC.STATE: [self.index],
                FC.TURBINE: np.arange(self.n_turbines),
                FC.POINT: np.arange(n_points),
                FC.XYH: np.arange(3),
            },
            data_vars={
                v: (self._tdata.dims[v], d[i, None]) for v, d in self._tdata.items()
            },
        )

        return results

    def calc_farm(self, *args, **kwargs):
        if not self.iterating:
            raise ValueError(f"calc_farm call is only allowed during iterations")

        return self.cur_farm_results

    def calc_points(self, farm_results, points):
        if not self.iterating:
            raise ValueError(f"calc_points call is only allowed during iterations")

        n_points = points.shape[1]

        plist, calc_pars = self._collect_point_models(ambient=self.ambient)
        if not plist.initialized:
            plist.initialize(self, self.verbosity)
        pvars = plist.output_point_vars(self)

        mdata = self.get_models_idata()
        mdata = MData(
            data={v: d[1] for v, d in mdata["data_vars"].items()},
            dims={v: d[0] for v, d in mdata["data_vars"].items()},
            loop_dims=[FC.STATE],
            name="mdata",
        )
        mdata = MData(
            data={
                v: d[self.states.counter, None] if mdata.dims[v][0] == FC.STATE else d
                for v, d in mdata.items()
            },
            dims={v: d for v, d in mdata.dims.items()},
            loop_dims=[FC.STATE],
            name="mdata",
        )

        fdata = FData(
            data={v: farm_results[v].to_numpy() for v in self.farm_vars},
            dims={v: (FC.STATE, FC.TURBINE) for v in self.farm_vars},
            loop_dims=[FC.STATE],
            name="fdata",
        )

        tdata = TData.from_points(
            points[0, None],
            data={v: np.zeros((1, n_points, 1), dtype=FC.DTYPE) for v in pvars},
            dims={v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in pvars},
            name="tdata",
        )

        pres = plist.calculate(self, mdata, fdata, tdata, parameters=calc_pars)
        pres = Dataset(
            coords={FC.STATE: self.states.index()},
            data_vars={
                v: ((FC.STATE, FC.TARGET, FC.TPOINT), d) for v, d in pres.items()
            },
        )

        # plist.finalize(self, self.verbosity)

        return pres
