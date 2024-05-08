from abc import abstractmethod
import numpy as np

from foxes.utils import all_subclasses
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
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

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
