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

    :group: algorithms.sequential.models

    """

    def __init__(self, algo, points=None):
        """
        Constructor.
        
        Parameters
        ----------
        algo: foxes.algorithms.Sequential
            The algorithm
        points: numpy.ndarray, optional
            The points of interest, shape: (n_states, n_points, 3)

        """
        self.algo = algo
        self.states = algo.states.states
        self.points = points

    def __iter__(self):
        """ Initialize the iterator """

        if not self.algo.initialized:
            self.algo.initialize()

        self._inds = self.states.index()
        self._weights = self.states.weights(self.algo)
        self._i = 0

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
            name="mdata"
        )

        self._fdata = Data(
            data={v: np.zeros((self.algo.n_states, self.algo.n_turbines), dtype=FC.DTYPE) for v in self.algo.farm_vars},
            dims={v: (FC.STATE, FC.TURBINE) for v in self.algo.farm_vars},
            loop_dims=[FC.STATE],
            name="fdata"
        )

        self._pdata = Data.from_points(self.points) if self.points is not None else None
        
        return self
    
    def __next__(self):
        """ Iterate to next state """

        if self._i < len(self._inds):

            self.algo.states._size = 1
            self.algo.states._indx = self.index
            self.algo.states._weight = self.weight

            mdata = Data(
                data={v: d[self._i, None] if self._mdata.dims[v][0] == FC.STATE else d
                      for v, d in self._mdata.items()},
                dims={v: d for v, d in self._mdata.dims.items()},
                loop_dims=[FC.STATE],
                name="mdata"
            )

            fdata = Data(
                data={v: d[self._i, None] for v, d in self._fdata.items()},
                dims={v: d for v, d in self._fdata.dims.items()},
                loop_dims=[FC.STATE],
                name="fdata"
            )
            
            fres = self._mlist.calculate(self.algo, mdata, fdata, parameters=self._calc_pars)
            fres[FV.WEIGHT] = self.weight[None, :]
            for v, d in fres.items():
                self._fdata[v][self._i] = d[0]

            self._i += 1

            return fres
        
        else:

            del self._i, self._mdata

            self.algo.states._size = len(self._inds)
            self.algo.states._indx = self._inds
            self.algo.states._weight = self._weights

            raise StopIteration

    @property
    def counter(self):
        """
        The current index counter
        
        Returns
        -------
        i: int
            The current index counter

        """
        return self._i

    @property
    def index(self):
        """
        The current index
        
        Returns
        -------
        indx: int
            The current index

        """
        return self._inds[self._i]

    @property
    def weight(self):
        """
        The current weight array
        
        Returns
        -------
        w: numpy.ndarray
            The current weight array, shape: (n_turbines,)

        """
        return self._weights[self._i]
    
    @property
    def mdata(self):
        """
        Get the current model data
        
        Returns
        -------
        d: foxes.core.Data
            The current model data

        """
        return self._mdata

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
        return self._pdata
    
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
            coords={FC.STATE: self._inds, FC.TURBINE: np.arange(self.algo.n_turbines)},
            data_vars={v: (self._fdata.dims[v], d) for v, d in self._fdata.items()}
        )

        results[FC.TNAME] = ((FC.TURBINE,), self.algo.farm.turbine_names)
        if FV.ORDER in results:
            results[FV.ORDER] = results[FV.ORDER].astype(FC.ITYPE)

        return results
