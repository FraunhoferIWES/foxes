import numpy as np

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC


class WSLinear(WakeSuperposition):
    """
    Linear supersposition of wind deficit results

    Attributes
    ----------
    scale_amb: bool
        Flag for scaling wind deficit with ambient wind speed
        instead of waked wind speed
    lim_low: float
        Lower limit of the final waked wind speed
    lim_high: float
        Upper limit of the final waked wind speed

    :group: models.wake_superpositions

    """

    def __init__(self, scale_amb=False, lim_low=None, lim_high=None):
        """
        Constructor.

        Parameters
        ----------
        scale_amb: bool
            Flag for scaling wind deficit with ambient wind speed
            instead of waked wind speed
        lim_low: float
            Lower limit of the final waked wind speed
        lim_high: float
            Upper limit of the final waked wind speed

        """
        super().__init__()

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
        Add a wake delta to previous wake deltas,
        at rotor points.

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
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(sr_sel):
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_ROTOR,
                lookup="w",
                algo=algo,
                fdata=fdata,
                pdata=pdata,
                downwind_index=downwind_index,
            )[sr_sel, None]
            
            wake_delta[sr_sel] += scale * wake_model_result

        return wake_delta

    def add_at_points(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        downwind_index,
        sp_sel,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas,
        at points of interest.

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
        sp_sel: numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape: 
            (n_states, n_points, ...)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected rotors,
            shape: (n_sp_sel, ...)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape: 
            (n_states, n_points, ...)

        """
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(sp_sel):
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_POINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                pdata=pdata,
                downwind_index=downwind_index,
            )[sp_sel]

            wake_delta[sp_sel] += scale * wake_model_result

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
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
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
        w = wake_delta
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w
