import numpy as np

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC


class WSProduct(WakeSuperposition):
    """
    Product supersposition of wind deficit results

    This is based on the idea that the dimensionless
    wind deficit should be rescaled with the wake
    corrected wind field, rather than the rotor
    equivalent wind speed.

    Source: https://arxiv.org/pdf/2010.03873.pdf
            Equation (8)

    Attributes
    ----------
    lim_low: float
        Lower limit of the final waked wind speed
    lim_high: float
        Upper limit of the final waked wind speed

    :group: models.wake_superpositions

    """

    def __init__(self, lim_low=None, lim_high=None):
        """
        Constructor.

        Parameters
        ----------
        lim_low: float
            Lower limit of the final waked wind speed
        lim_high: float
            Upper limit of the final waked wind speed

        """
        super().__init__()

        self.lim_low = lim_low
        self.lim_high = lim_high

    def input_farm_vars(self, algo):
        """
        The variables which are needed for running
        the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        input_vars: list of str
            The input variable names

        """
        return [FV.AMB_REWS] if self.scale_amb else [FV.REWS]

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
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(sel_sp):
            if np.max(np.abs(wake_delta)) < 1e-14:
                wake_delta[:] = 1

            wake_delta[sel_sp] *= 1 + wake_model_result

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
        w = amb_results * (wake_delta - 1)
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w
