import numpy as np

from foxes.core import WakeSuperposition
import foxes.variables as FV


class TIQuadratic(WakeSuperposition):
    """
    Quadratic wake superposition for TI.

    Attributes
    ----------
    superp_to_amb: str
        The method for combining ambient with wake deltas:
        linear or quadratic

    :group: models.wake_superpositions

    """

    def __init__(self, superp_to_amb="quadratic"):
        """
        Constructor.

        Parameters
        ----------
        superp_to_amb: str
            The method for combining ambient with wake deltas:
            linear or quadratic

        """
        super().__init__()
        self.superp_to_amb = superp_to_amb

    def __repr__(self):
        return f"{type(self).__name__}(superp_to_amb={self.superp_to_amb})"

    def add_wake(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas,
        at rotor points.

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
        st_sel: numpy.ndarray of bool
            The selection of targets, shape: (n_states, n_targets)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, ...)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        if variable != FV.TI:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wake variable {FV.TI}, got {variable}"
            )

        wake_delta[st_sel] += wake_model_result**2
        return wake_delta

    def calc_final_wake_delta(
        self,
        algo,
        mdata,
        fdata,
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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        variable: str
            The variable name for which the wake deltas applies
        amb_results: numpy.ndarray
            The ambient results at targets,
            shape: (n_states, n_targets, n_tpoints)
        wake_delta: numpy.ndarray
            The wake deltas at targets, shape:
            (n_states, n_targets, n_tpoints)

        Returns
        -------
        final_wake_delta: numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        # linear superposition to ambient:
        if self.superp_to_amb == "linear":
            return np.sqrt(wake_delta)

        # quadratic superposition to ambient:
        elif self.superp_to_amb == "quadratic":
            return np.sqrt(wake_delta + amb_results**2) - amb_results

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown superp_to_amb = '{self.superp_to_amb}', valid choices: linear, quadratic"
            )
