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

    def __repr__(self):
        a = f"scale_amb={self.scale_amb}, lim_low={self.lim_low}, lim_high={self.lim_high}"
        return f"{type(self).__name__}({a})"

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
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(st_sel):
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )[st_sel, None]

            wake_delta[st_sel] += scale * wake_model_result

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
        w = wake_delta
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w


class WSLinearLocal(WakeSuperposition):
    """
    Local linear supersposition of wind deficit results

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

    def __repr__(self):
        a = f"lim_low={self.lim_low}, lim_high={self.lim_high}"
        return f"{type(self).__name__}({a})"

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
        return []

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
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(st_sel):
            wake_delta[st_sel] += wake_model_result

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
        w = wake_delta * amb_results
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w
