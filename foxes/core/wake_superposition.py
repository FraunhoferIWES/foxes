from abc import abstractmethod

from .model import Model


class WakeSuperposition(Model):
    """
    Abstract base class for wake superposition models.

    Note that it is a matter of the wake model
    if superposition models are used, or if the
    wake model computes the total wake result by
    other means.

    :group: core

    """

    @abstractmethod
    def add_at_rotors(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        downwind_index,
        sr_sel,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data at rotor points
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        sr_sel: numpy.ndarray of bool
            The selection of rotors, shape: (n_states, n_rotors)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape: 
            (n_states, n_rotors, n_rpoints, ...)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected rotors,
            shape: (n_sr_sel, n_rpoints, ...)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape: 
            (n_states, n_rotors, n_rpoints, ...)

        """
        pass

    @abstractmethod
    def calc_final_wake_delta(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        variable,
        amb_results,
        wake_delta,
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        variable: str
            The variable name for which the wake deltas applies
        amb_results: numpy.ndarray
            The ambient results, shape: (n_states, n_points)
        wake_delta: numpy.ndarray
            The wake deltas, shape: (n_states, n_points)

        Returns
        -------
        final_wake_delta: numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape: (n_states, n_points)

        """
        pass
