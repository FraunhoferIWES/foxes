import numpy as np

from foxes.core import WakeSuperposition


class TISuperposition(WakeSuperposition):
    """
    A collection of superpositions for TI.

    Parameters
    ----------
    ti_superp : str
        The method choice: linear, quadratic, max
    superp_to_amb : str
        The method for combining ambient with wake deltas:
        linear or quadratic

    Attributes
    ----------
    ti_superp : str
        The method choice: linear, quadratic, max
    superp_to_amb : str
        The method for combining ambient with wake deltas:
        linear or quadratic

    """

    def __init__(self, ti_superp, superp_to_amb="quadratic"):
        super().__init__()

        self.ti_superp = ti_superp
        self.superp_to_amb = superp_to_amb

    def calc_wakes_plus_wake(
        self,
        algo,
        mdata,
        fdata,
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
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        sel_sp : numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable : str
            The variable name for which the wake deltas applies
        wake_delta : numpy.ndarray
            The original wake deltas, shape: (n_states, n_points)
        wake_model_result : numpy.ndarray
            The new wake deltas of the selected points,
            shape: (n_sel_sp,)

        Returns
        -------
        wdelta : numpy.ndarray
            The updated wake deltas, shape: (n_states, n_points)

        """
        # superposition of every turbines efect at each target point
        # linear ti delta:
        if self.ti_superp == "linear":
            wake_delta[sel_sp] += wake_model_result

        # quadratic ti delta:
        elif self.ti_superp == "quadratic":
            wake_delta[sel_sp] += wake_model_result**2

        # max ti delta:
        elif self.ti_superp == "max":
            wake_delta[sel_sp] = np.maximum(wake_model_result, wake_delta[sel_sp])

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, max"
            )

        return wake_delta

    def calc_final_wake_delta(
        self, algo, mdata, fdata, variable, amb_results, wake_delta
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        variable : str
            The variable name for which the wake deltas applies
        amb_results : numpy.ndarray
            The ambient results, shape: (n_states, n_points)
        wake_delta : numpy.ndarray
            The wake deltas, shape: (n_states, n_points)

        Returns
        -------
        final_wake_delta : numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape: (n_states, n_points)

        """
        # linear superposition to ambient:
        if self.superp_to_amb == "linear":

            if self.ti_superp == "linear" or self.ti_superp == "max":
                return wake_delta
            elif self.ti_superp == "quadratic":
                return np.sqrt(wake_delta)
            else:
                raise ValueError(
                    f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, max"
                )

        # quadratic superposition to ambient:
        elif self.superp_to_amb == "quadratic":

            if self.ti_superp == "linear" or self.ti_superp == "max":
                return np.sqrt(wake_delta**2 + amb_results**2) - amb_results
            elif self.ti_superp == "quadratic":
                return np.sqrt(wake_delta + amb_results**2) - amb_results
            else:
                raise ValueError(
                    f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, max"
                )

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown superp_to_amb = '{self.superp_to_amb}', valid choices: linear, quadratic"
            )
