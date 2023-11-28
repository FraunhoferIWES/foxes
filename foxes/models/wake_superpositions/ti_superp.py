import numpy as np

from foxes.core import WakeSuperposition


class TISuperposition(WakeSuperposition):
    """
    A collection of superpositions for TI.

    Attributes
    ----------
    ti_superp: str
        The method choice: linear, quadratic, power_N, max
    superp_to_amb: str
        The method for combining ambient with wake deltas:
        linear or quadratic

    :group: models.wake_superpositions

    """

    def __init__(self, ti_superp, superp_to_amb="quadratic"):
        """
        Constructor.

        Parameters
        ----------
        ti_superp: str
            The method choice: linear, quadratic, power_N, max
        superp_to_amb: str
            The method for combining ambient with wake deltas:
            linear or quadratic

        """
        super().__init__()

        self.ti_superp = ti_superp
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

        # superposition of every turbines efect at each target point
        # linear ti delta:
        if self.ti_superp == "linear":
            wake_delta[sel_sp] += wake_model_result

        # quadratic ti delta:
        elif self.ti_superp == "quadratic":
            wake_delta[sel_sp] += wake_model_result**2

        # power_N delta:
        elif len(self.ti_superp) > 6 and self.ti_superp[:6] == "power_":
            N = int(self.ti_superp[6:])
            wake_delta[sel_sp] += wake_model_result**N

        # max ti delta:
        elif self.ti_superp == "max":
            wake_delta[sel_sp] = np.maximum(wake_model_result, wake_delta[sel_sp])

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, power_N, max"
            )

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
            if self.ti_superp == "linear" or self.ti_superp == "max":
                return wake_delta
            elif self.ti_superp == "quadratic":
                return np.sqrt(wake_delta)
            elif len(self.ti_superp) > 6 and self.ti_superp[:6] == "power_":
                N = int(self.ti_superp[6:])
                return wake_delta ** (1 / N)
            else:
                raise ValueError(
                    f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, power_N, max"
                )

        # quadratic superposition to ambient:
        elif self.superp_to_amb == "quadratic":
            if self.ti_superp == "linear" or self.ti_superp == "max":
                return np.sqrt(wake_delta**2 + amb_results**2) - amb_results
            elif self.ti_superp == "quadratic":
                return np.sqrt(wake_delta + amb_results**2) - amb_results
            elif len(self.ti_superp) > 6 and self.ti_superp[:6] == "power_":
                N = int(self.ti_superp[6:])
                return np.sqrt(wake_delta ** (2 / N) + amb_results**2) - amb_results
            else:
                raise ValueError(
                    f"Unknown ti_superp = '{self.ti_superp}', valid choices: linear, quadratic, power_N, max"
                )

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown superp_to_amb = '{self.superp_to_amb}', valid choices: linear, quadratic"
            )
