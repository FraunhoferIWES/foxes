from abc import abstractmethod

from foxes.utils import new_instance

from .model import Model


class PartialWakesModel(Model):
    """
    Abstract base class for partial wakes models.

    Partial wakes models compute wake effects
    for rotor effective quantities.

    Attributes
    ----------
    wake_models: list of foxes.core.WakeModel
        The wake model selection
    wake_frame: foxes.core.WakeFrame, optional
        The wake frame

    :group: core

    """

    def check_wmodel(self, wmodel, error=True):
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel: foxes.core.WakeModel
            The wake model to be tested
        error: bool
            Flag for raising TypeError

        Returns
        -------
        chk: bool
            True if wake model is compatible

        """
        return True

    @abstractmethod
    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points, and their
        weights.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        rpoints: numpy.ndarray
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights: numpy.ndarray
            The target point weights, shape: (n_tpoints,)

        """
        pass

    def new_wake_deltas(self, algo, mdata, fdata, tdata, wmodel):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The target point data
        wmodel: foxes.core.WakeModel
            The wake model

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_tpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
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
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

    @abstractmethod
    def finalize_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        amb_res,
        rpoint_weights,
        wake_deltas,
        wmodel,
        downwind_index,
    ):
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.Data
            The target point data
        amb_res: dict
            The ambient results at the target points
            of all rotors. Key: variable name, value
            np.ndarray of shape:
            (n_states, n_turbines, n_rotor_points)
        rpoint_weights: numpy.ndarray
            The rotor point weights, shape: (n_rotor_points,)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: np.ndarray of shape
            (n_states, n_turbines, n_tpoints)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        Returns
        -------
        final_wake_deltas: dict
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        pass

    @classmethod
    def new(cls, pwakes_type, *args, **kwargs):
        """
        Run-time partial wakes model factory.

        Parameters
        ----------
        pwakes_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, pwakes_type, *args, **kwargs)
