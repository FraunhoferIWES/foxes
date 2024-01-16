import numpy as np

from foxes.core import WakeSuperposition
import foxes.variables as FV

class TILinear(WakeSuperposition):
    """
    Linear wake superposition for TI.

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

    def calc_wakes_plus_wake(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        sel_sp,
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
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        sel_sp: numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape: (n_states, n_points)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected points,
            shape: (n_sel_sp,)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape: (n_states, n_points)

        """
        if variable != FV.TI:
            raise ValueError(f"Superposition '{self.name}': Expecting wake variable {FV.TI}, got {variable}")
        
        wake_delta[sel_sp] += wake_model_result
        return wake_delta

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
        # linear superposition to ambient:
        if self.superp_to_amb == "linear":
            return wake_delta

        # quadratic superposition to ambient:
        elif self.superp_to_amb == "quadratic":
            return np.sqrt(wake_delta**2 + amb_results**2) - amb_results

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown superp_to_amb = '{self.superp_to_amb}', valid choices: linear, quadratic"
            )
