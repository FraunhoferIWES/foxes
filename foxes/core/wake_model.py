from abc import abstractmethod
import numpy as np

from foxes.utils import all_subclasses, Factory
import foxes.variables as FV
import foxes.constants as FC

from .model import Model


class WakeModel(Model):
    """
    Abstract base class for wake models.

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
        return True

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
        return {FV.WS: np.zeros_like(tdata[FC.TARGETS][..., 0])}

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
            in the downwnd order
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
        amb_results,
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
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
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

        if wmodel_type is None:
            return None

        allc = all_subclasses(cls)
        found = wmodel_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == wmodel_type:
                    return scls(*args, **kwargs)

        else:
            estr = (
                "Wake model type '{}' is not defined, available types are \n {}".format(
                    wmodel_type, sorted([i.__name__ for i in allc])
                )
            )
            raise KeyError(estr)


class TurbineInductionModel(WakeModel):
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
        if self._k is not None:
            setattr(self, self.k_var, self._k)
        elif self._ka is not None or self._kb is not None:
            if self.ti_var == FV.TI and ti is not None:
                pass
            elif self.ti_var == FV.AMB_TI and amb_ti is not None:
                ti = amb_ti
            else:
                ti = self.get_data(self.ti_var, *args, lookup=lookup_ti, **kwargs)
            kb = 0 if self._kb is None else self._kb
            setattr(self, self.k_var, self._ka * ti + kb)

        k = self.get_data(self.k_var, *args, lookup=lookup_k, **kwargs)
        setattr(self, self.k_var, None)
        return k
