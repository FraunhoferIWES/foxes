import numpy as np
from xarray import Dataset

from foxes.algorithms.downwind.downwind import Downwind
import foxes.constants as FC
from foxes.core.data import Data

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
            mbook, 
            farm, 
            states, 
            *args, 
            ambient=False, 
            calc_pars={},
            chunks={FC.STATE: None, FC.POINT: 10000},
            **kwargs,
        ):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.ModelBook
            The model book
        farm: foxes.WindFarm
            The wind farm
        states: foxes.core.States
            The ambient states
        args: tuple, optional
            Arguments for Downwind
        ambient: bool
            Flag for ambient calculation
        calc_pars: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        kwargs: dict, optional
            Keyword arguments for Downwind

        """
        super().__init__(
            mbook,
            farm,
            mdls.DummyStates(states),
            *args, 
            chunks=chunks,
            **kwargs
        )
        self.ambient = ambient
        self.calc_pars = calc_pars
    
    def iter(self, *args, **kwargs):
        """
        Get a cusomized iterator
        
        Parameters
        ----------
        args: tuple, optional
            Additional arguments for the constructor
        kwargs: dict, optional
            Additional arguments for the constructor
        
        """
        return iter(mdls.SequentialIter(self, *args, **kwargs))

    def __iter__(self):
        """ Get the default iterator """
        return self.iter()

    def calc_farm(self, *args, **kwargs):
        raise NotImplementedError
    
    def calc_points(self, farm_results, points):

        if self.states.size() != 1:
            raise ValueError(f"Expecting states of size 1, found {self.states.size()}. Maybe calc_points was called not during sequential iteration?")

        n_points = points.shape[1]

        plist, calc_pars = self._collect_point_models(ambient=self.ambient)
        pvars = plist.output_point_vars(self)

        mdata = self.get_models_idata()
        mdata = Data(
            data={v: d[1] for v, d in mdata["data_vars"].items()},
            dims={v: d[0] for v, d in mdata["data_vars"].items()},
            loop_dims=[FC.STATE],
            name="mdata",
        )
        mdata = Data(
            data={v: d[self.states.counter, None] if mdata.dims[v][0] == FC.STATE else d
                    for v, d in mdata.items()},
            dims={v: d for v, d in mdata.dims.items()},
            loop_dims=[FC.STATE],
            name="mdata",
        )
        
        fdata = Data(
            data={v: farm_results[v].to_numpy() for v in self.farm_vars},
            dims={v: (FC.STATE, FC.TURBINE) for v in self.farm_vars},
            loop_dims=[FC.STATE],
            name="fdata",
        )
        
        pdata = Data.from_points(
            points[0, None],
            data={v: np.zeros((1, n_points), dtype=FC.DTYPE) for v in pvars},
            dims={v: (FC.STATE, FC.POINT) for v in pvars},
            name="pdata",
        )

        pres = plist.calculate(self, mdata, fdata, pdata, parameters=calc_pars)
        pres = Dataset(
            coords={FC.STATE: [self.states.index], FC.POINT: np.arange(n_points)},
            data_vars={v: ((FC.STATE, FC.POINT), d) for v, d in pres.items()}
        )
    
        #plist.finalize(self, self.verbosity)

        return pres