from abc import abstractmethod

from .model import Model


class WakeModel(Model):
    """
    Abstract base class for wake models.
    """

    @abstractmethod
    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        """
        Initialize wake delta storage.

        They are added on the fly to the wake_deltas dict.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        n_points : int
            The number of wake evaluation points
        wake_deltas : dict
            The wake deltas storage, add wake deltas
            on the fly. Keys: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        pass

    @abstractmethod
    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_coos, wake_deltas
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        wake_deltas : dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        pass

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        amb_results : dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas : dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """
        pass
