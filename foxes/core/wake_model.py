from abc import abstractmethod

from foxes.utils import new_instance
import foxes.variables as FV

from .model import Model
from .wake_superposition import WindVectorWakeSuperposition


class WakeModel(Model):
    """
    Abstract base class for wake models.

    :group: core

    """
    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self._has_uv = False

    @property
    def affects_downwind(self):
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        dwnd: bool
            Flag for downwind effects by this model

        """
        return True
    
    @property
    def has_uv(self):
        """
        This model uses wind vector data
        
        Returns
        -------
        hasuv: bool
            Flag for wind vector data
        
        """
        return self._has_uv

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if algo.wake_deflection.has_uv:
            self._has_uv = True
        super().initialize(algo, verbosity, force)

    @abstractmethod
    def new_wake_deltas(self, algo, mdata, fdata, tdata):
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        pass

    @abstractmethod
    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        pass

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        pass

    @classmethod
    def new(cls, wmodel_type, *args, **kwargs):
        """
        Run-time wake model factory.

        Parameters
        ----------
        wmodel_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, wmodel_type, *args, **kwargs)


class SingleTurbineWakeModel(WakeModel):
    """
    Abstract base class for wake models that represent
    a single turbine wake

    Single turbine wake models depend on superposition models.

    Attributes
    ----------
    wind_superposition: str
        The wind superposition model name (vector or compenent model),
        will be looked up in model book
    other_superpositions: dict
        The superpositions for other than (ws, wd) variables. 
        Key: variable name str, value: The wake superposition 
        model name, will be looked up in model book
    vec_superp: foxes.core.WindVectorWakeSuperposition or None
        The wind vector wake superposition model
    superp: dict
        The superposition dict, key: variable name str,
        value: `foxes.core.WakeSuperposition`

    :group: models.wake_models

    """

    def __init__(self, wind_superposition=None, other_superpositions={}):
        """
        Constructor.

        Parameters
        ----------
        wind_superposition: str, optional
            The wind superposition model name (vector or compenent model),
            will be looked up in model book
        other_superpositions: dict
            The superpositions for other than (ws, wd) variables. 
            Key: variable name str, value: The wake superposition 
            model name, will be looked up in model book

        """
        super().__init__()
        self.wind_superposition = wind_superposition
        self.other_superpositions = other_superpositions
        self.vec_superp = None
        self.superp = {}

        for v in [FV.WS, FV.WD]:
            assert v not in other_superpositions, f"Wake model '{self.name}': Found variable '{v}' among 'other_superposition' keyword, use 'wind_superposition' instead"

        self.__has_vector_superp = False

    @property
    def has_vector_wind_superp(self):
        """
        This model uses a wind vector superposition
        
        Returns
        -------
        hasv: bool
            Flag for wind vector superposition
        
        """
        return self.__has_vector_superp
    
    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        w = [self.vec_superp] if self.vec_superp is not None else []
        return w + list(self.superp.values())

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        self.superp = {
            v: algo.mbook.wake_superpositions[s] for v, s in self.other_superpositions.items()
        }

        if self.wind_superposition is not None:
            self.vec_superp = algo.mbook.wake_superpositions[self.wind_superposition]
            self.__has_vector_superp = isinstance(self.vec_superp, WindVectorWakeSuperposition)
            if self.__has_vector_superp:
                self._has_uv = True
            else:
                self.superp[FV.WS] = self.vec_superp
                self.vec_superp = None
            
        super().initialize(algo, verbosity, force)

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        for v in wake_deltas.keys():
            if v != FV.UV:
                try:
                    wake_deltas[v] = self.superp[v].calc_final_wake_delta(
                        algo, mdata, fdata, tdata, v, wake_deltas[v]
                    )
                except KeyError:
                    raise KeyError(f"Wake model '{self.name}': Variable '{v}' appears to be modified, missing superposition model")

        if FV.UV in wake_deltas:
            assert self.has_vector_wind_superp, f"{self.name}: Expecting wind vector superposition, got '{self.wind_superposition}'"
            dws, dwd = self.vec_superp.calc_final_wake_delta_uv(
                    algo, mdata, fdata, tdata, wake_deltas.pop(FV.UV)
                )

            wake_deltas[FV.WS] = dws
            wake_deltas[FV.WD] = dwd
            

class TurbineInductionModel(SingleTurbineWakeModel):
    """
    Abstract base class for turbine induction models.

    :group: core

    """

    @property
    def affects_downwind(self):
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        dwnd: bool
            Flag for downwind effects by this model

        """
        return False

    @classmethod
    def new(cls, induction_type, *args, **kwargs):
        """
        Run-time turbine induction model factory.

        Parameters
        ----------
        induction_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, induction_type, *args, **kwargs)


class WakeK(Model):
    """
    Handler for the wake growth parameter k

    Attributes
    ----------
    k_var: str
        The name of the k variable
    ti_var: str
        The name of the TI variable

    :group: core

    """

    def __init__(
        self,
        k=None,
        ka=None,
        kb=None,
        k_var=FV.K,
        ti_var=FV.TI,
    ):
        """
        Constructor.

        Parameters
        ----------
        k: float, optional
            The k value
        ka: float, optional
            The ka value in k = ka * TI + kb
        kb: float, optional
            The kb value in k = ka * TI + kb
        k_var: str
            The name of the k variable
        ti_var: str
            The name of the TI variable

        """
        super().__init__()
        self._k = k
        self._ka = ka
        self._kb = kb
        self.k_var = k_var
        self.ti_var = ti_var

        if k is not None and (ka is not None or kb is not None):
            raise ValueError("Got 'k' and also ('ka' or 'kb') as non-None parameters")
        elif k is None and kb is not None and (ka is None or ka == 0):
            raise ValueError(f"Got k={k}, ka={ka}, kb={kb}, use k={kb} instead")

        setattr(self, self.k_var, None)

    def repr(self):
        """
        Provides the representative string

        Returns
        -------
        s: str
            The representative string

        """
        if self._k is not None:
            s = f"{self.k_var}={self._k}"
        elif self._ka is not None or self._kb is not None:
            s = f"{self.k_var}={self._ka}*{self.ti_var}"
            if self._kb is not None and self._kb > 0:
                s += f"+{self._kb}"
        else:
            s = f"k_var={self.k_var}"
        return s

    @property
    def is_kTI(self):
        """Flag for ka != 0"""
        return self._ka is not None and self._ka != 0

    @property
    def all_none(self):
        """Flag for k=ka=kb=None"""
        return self._k is None and self._ka is None and self._kb is None

    @property
    def use_amb_ti(self):
        """Flag for using ambient ti"""
        return self.ti_var in FV.amb2var

    def __call__(
        self,
        *args,
        lookup_ti="w",
        lookup_k="sw",
        ti=None,
        amb_ti=None,
        **kwargs,
    ):
        """
        Gets the k value

        Parameters
        ----------
        args: tuple, optional
            Arguments for get_data
        lookup_ti: str
            The ti lookup order for get_data
        lookup_k: str
            The k lookup order for get_data
        ti: numpy.ndarray, optional
            ti data in the requested target shape,
            if known
        amb_ti: numpy.ndarray, optional
            Ambient ti data in the requested target shape,
            if known
        kwargs: dict, optional
            Arguments for get_data

        Returns
        -------
        k: numpy.ndarray
            The k array as returned by get_data

        """
        sel = kwargs.pop("selection", None)
        setattr(self, self.k_var, self._k)
        if self._ka is not None or self._kb is not None:
            if self.ti_var == FV.TI and ti is not None:
                pass
            elif self.ti_var == FV.AMB_TI and amb_ti is not None:
                ti = amb_ti
            else:
                ti = self.get_data(self.ti_var, *args, lookup=lookup_ti, **kwargs)
            kb = 0 if self._kb is None else self._kb
            setattr(self, self.k_var, self._ka * ti + kb)

        k = self.get_data(self.k_var, *args, lookup=lookup_k, selection=sel, **kwargs)
        setattr(self, self.k_var, None)
        return k
