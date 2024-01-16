import numpy as np

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC

class WSPow(WakeSuperposition):
    """
    Power supersposition of wind deficit results

    Attributes
    ----------
    pow: float
        The power to which to take the wake results
    scale_amb: bool
        Flag for scaling wind deficit with ambient wind speed
        instead of waked wind speed
    lim_low: float
        Lower limit of the final waked wind speed
    lim_high: float
        Upper limit of the final waked wind speed

    :group: models.wake_superpositions

    """

    def __init__(self, pow, scale_amb=False, lim_low=None, lim_high=None):
        """
        Constructor.

        Parameters
        ----------
        pow: float
            The power to which to take the wake results
        scale_amb: bool
            Flag for scaling wind deficit with ambient wind speed
            instead of waked wind speed
        lim_low: float
            Lower limit of the final waked wind speed
        lim_high: float
            Upper limit of the final waked wind speed

        """
        super().__init__()

        self.pow = pow
        self.scale_amb = scale_amb
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
            raise ValueError(f"Superposition '{self.name}': Expecting wind speed variable, got {variable}")
        
        if np.any(sel_sp):
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_POINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                pdata=pdata,
                upcast=True,
                states_source_turbine=states_source_turbine,
            )[sel_sp]

            wake_delta[sel_sp] += np.abs(scale * wake_model_result)**self.pow

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
        w = -wake_delta**(1/self.pow)
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w
