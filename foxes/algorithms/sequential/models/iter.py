import numpy as np
from xarray import Dataset

from foxes.core import Data
import foxes.constants as FC
import foxes.variables as FV

class SequentialIter:
    """
    Iterator for the sequential algorithm
    
    Attributes
    ----------
    algo: foxes.algorithms.Sequential
        The algorithm
    states: foxes.core.States
        The underlying states
    points: numpy.ndarray
        The points of interest, shape: (n_states, n_points, 3)
    store: bool
        Flag for storing state results

    :group: algorithms.sequential.models

    """

    def __init__(self, algo, points=None, store=True):
        """
        Constructor.
        
        Parameters
        ----------
        algo: foxes.algorithms.Sequential
            The algorithm
        points: numpy.ndarray, optional
            The points of interest, shape: (n_states, n_points, 3)
        store: bool
            Flag for storing state results

        """
        self.algo = algo
        self.states = algo.states.states
        self.points = points
        self.store = store

        self._i = None

    def __iter__(self):
        """ Initialize the iterator """

        if self._i is None:

            if not self.algo.initialized:
                self.algo.initialize()

            self._inds = self.states.index()
            self._weights = self.states.weights(self.algo)
            self._i = 0
            self._counter = 0

            self._mlist, self._calc_pars = self.algo._collect_farm_models(self.algo.calc_pars, self.algo.ambient)
            self._mdata = self.algo.get_models_idata()
            if self.algo.verbosity > 0:
                s = "\n".join([f'  {v}: {d[0]} {d[1].dtype}, shape {d[1].shape}' 
                            for v, d in self._mdata['data_vars'].items()])
                print("\nInput data:\n")
                print(s, "\n")
                print(f"Output farm variables:", ", ".join(self.algo.farm_vars))
                print()

            self._mdata = Data(
                data={v: d[1] for v, d in self._mdata["data_vars"].items()},
                dims={v: d[0] for v, d in self._mdata["data_vars"].items()},
                loop_dims=[FC.STATE],
                name="mdata",
            )

            if self.store:
                self._fdata = Data(
                    data={v: np.zeros((self.algo.n_states, self.algo.n_turbines), dtype=FC.DTYPE) for v in self.algo.farm_vars},
                    dims={v: (FC.STATE, FC.TURBINE) for v in self.algo.farm_vars},
                    loop_dims=[FC.STATE],
                    name="fdata",
                )

            if self.points is not None:

                self._plist, self._calc_pars_p = self.algo._collect_point_models(ambient=self.algo.ambient)
                self._pvars = self._plist.output_point_vars(self.algo)
                self.algo.print(f"\nOutput point variables:", ", ".join(self._pvars), "\n")

                if self.store:
                    n_points = self.points.shape[1]
                    self._pdata = Data.from_points(
                        self.points,
                        data={v: np.zeros((self.algo.n_states, n_points), dtype=FC.DTYPE) for v in self._pvars},
                        dims={v: (FC.STATE, FC.POINT) for v in self._pvars},
                        name="pdata",
                    )
        
        return self
    
    def __next__(self):
        """ Run calculation for current step, then iterate to next """

        if self._i < len(self._inds):

            self._counter = self._i
            self.algo.states._size = 1
            self.algo.states._indx = self._inds[self._i]
            self.algo.states._weight = self._weights[self._i]

            mdata = Data(
                data={v: d[self._i, None] if self._mdata.dims[v][0] == FC.STATE else d
                      for v, d in self._mdata.items()},
                dims={v: d for v, d in self._mdata.dims.items()},
                loop_dims=[FC.STATE],
                name="mdata",
            )

            fdata = Data(
                data={v: np.zeros((1, self.algo.n_turbines), dtype=FC.DTYPE) for v in self.algo.farm_vars},
                dims={v: (FC.STATE, FC.TURBINE) for v in self.algo.farm_vars},
                loop_dims=[FC.STATE],
                name="fdata",
            )
            
            fres = self._mlist.calculate(self.algo, mdata, fdata, parameters=self._calc_pars)
            fres[FV.WEIGHT] = self.weight[None, :]

            if self.store:
                for v, d in fres.items():
                    self._fdata[v][self._i] = d[0]
            else:
                self._fdata = fdata
            
            if self.points is None:
                self._i += 1
                return fres
            
            else:
                n_points = self.points.shape[1]
                pdata = Data.from_points(
                    self.points[self.counter, None],
                    data={v: np.zeros((1, n_points), dtype=FC.DTYPE) for v in self._pvars},
                    dims={v: (FC.STATE, FC.POINT) for v in self._pvars},
                    name="pdata",
                )

                pres = self._plist.calculate(self.algo, mdata, fdata, pdata, parameters=self._calc_pars_p)
                if self.store:
                    for v, d in pres.items():
                        self._pdata[v][self._i] = d[0]
                else:
                    self._pdata = pdata

                self._i += 1
                return fres, pres
        
        else:

            del self._mdata

            self._i = None
            self.algo.states._size = len(self._inds)
            self.algo.states._indx = self._inds
            self.algo.states._weight = self._weights

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
        return self._counter if self._i is not None else None

    @property
    def index(self):
        """
        The current index
        
        Returns
        -------
        indx: int
            The current index

        """
        return self.algo.states._indx if self._i is not None else None

    @property
    def weight(self):
        """
        The current weight array
        
        Returns
        -------
        w: numpy.ndarray
            The current weight array, shape: (n_turbines,)

        """
        return self.algo.states._weight if self._i is not None else None
    
    @property
    def mdata(self):
        """
        Get the current model data
        
        Returns
        -------
        d: foxes.core.Data
            The current model data

        """
        return self._mdata if self._i is not None else None

    @property
    def fdata(self):
        """
        Get the current farm data
        
        Returns
        -------
        d: foxes.core.Data
            The current farm data

        """
        return self._fdata

    @property
    def pdata(self):
        """
        Get the current point data
        
        Returns
        -------
        d: foxes.core.Data
            The current point data

        """
        return self._pdata if self.points is not None and self._i is not None else None
    
    @property
    def farm_results(self):
        """
        The overall farm results
        
        Returns
        -------
        results: xarray.Dataset
            The overall farm results

        """

        if not self.store:
            raise ValueError(f"farm_results not stored, maybe you were looking for cur_farm_results?")

        results = Dataset(
            coords={FC.STATE: self._inds, FC.TURBINE: np.arange(self.algo.n_turbines)},
            data_vars={v: (self._fdata.dims[v], d) for v, d in self._fdata.items()}
        )

        results[FC.TNAME] = ((FC.TURBINE,), self.algo.farm.turbine_names)
        if FV.ORDER in results:
            results[FV.ORDER] = results[FV.ORDER].astype(FC.ITYPE)

        return results

    @property
    def cur_farm_results(self):
        """
        The current farm results
        
        Returns
        -------
        results: xarray.Dataset
            The current farm results

        """

        i = self.counter if self.store else 0
        results = Dataset(
            coords={FC.STATE: [self.index], FC.TURBINE: np.arange(self.algo.n_turbines)},
            data_vars={v: (self._fdata.dims[v], d[i, None]) for v, d in self._fdata.items()}
        )

        results[FC.TNAME] = ((FC.TURBINE,), self.algo.farm.turbine_names)
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

        if not self.store:
            raise ValueError(f"point_results not stored, maybe you were looking for cur_point_results?")
        
        n_points = self.points.shape[1]
        results = Dataset(
            coords={
                FC.STATE: self._inds, 
                FC.TURBINE: np.arange(self.algo.n_turbines), 
                FC.POINT: np.arange(n_points),
                FC.XYH: np.arange(3),
            },
            data_vars={v: (self._pdata.dims[v], d) for v, d in self._pdata.items()},
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
        i = self.counter if self.store else 0
        
        results = Dataset(
            coords={
                FC.STATE: [self.index], 
                FC.TURBINE: np.arange(self.algo.n_turbines), 
                FC.POINT: np.arange(n_points),
                FC.XYH: np.arange(3),
            },
            data_vars={v: (self._pdata.dims[v], d[i, None]) for v, d in self._pdata.items()},
        )

        return results
    