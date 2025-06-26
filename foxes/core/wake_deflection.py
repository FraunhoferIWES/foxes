from abc import abstractmethod

from foxes.utils import new_instance
from .model import Model


class WakeDeflection(Model):
    """
    Abstract base class for wake deflection models.

    :group: core

    """

    @property
    def has_uv(self):
        """
        This model uses wind vector data

        Returns
        -------
        hasuv: bool
            Flag for wind vector data

        """
        return False

    @abstractmethod
    def calc_deflection(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        coos,
    ):
        """
        Calculates the wake deflection.

        This function optionally adds FC.WDEFL_ROT_ANGLE or
        FC.WDEFL_DWS_FACTOR to the tdata.

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
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        Returns
        -------
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        pass

    def get_yaw_alpha_seq(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
    ):
        """
        Computes sequential wind vector rotation angles.

        Wind vector rotation angles are computed at the
        current trace points due to a yawed rotor
        for sequential runs.

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
        x: numpy.ndarray
            The distance from the wake causing rotor
            for the first n_times subsequent time steps,
            shape: (n_times,)

        Returns
        -------
        alpha: numpy.ndarray
            The delta WD result at the x locations,
            shape: (n_times,)

        """
        raise NotImplementedError(
            f"Wake deflection '{self.name}' not implemented for sequential runs"
        )

    @classmethod
    def new(cls, wframe_type, *args, **kwargs):
        """
        Run-time wake deflection model factory.

        Parameters
        ----------
        wframe_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, wframe_type, *args, **kwargs)
