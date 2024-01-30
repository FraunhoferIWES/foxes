import numpy as np

from foxes.core import WakeModel
import foxes.variables as FV
import foxes.constants as FC

class WakeMirror(WakeModel):
    """
    A wake model wrapper that adds mirror turbines
    that model wake reflection from a horizontal plane,
    e.g. the ground

    Attributes
    ----------
    wmodel: foxes.core.WakeModel
        The original wake model
    heights: list of float
        The reflection heights

    :group: models.wake_models

    """
    def __init__(self, wmodel, heights):
        """
        Constructor.

        Parameters
        ----------
        wmodel: foxes.core.WakeModel
            The original wake model
        heights: list of float
            The reflection heights

        """
        super().__init__()
        self.wmodel = wmodel
        self.heights = heights
        self.name = self.name + "_" + wmodel.name

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.wmodel]
    
    def init_wake_deltas(self, algo, mdata, fdata, pdata, wake_deltas):
        """
        Initialize wake delta storage.

        They are added on the fly to the wake_deltas dict.

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
        wake_deltas: dict
            The wake deltas storage, add wake deltas
            on the fly. Keys: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        self.wmodel.init_wake_deltas(algo, mdata, fdata, pdata, wake_deltas)

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_coos,
        wake_deltas,
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

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
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        stsel = (np.arange(algo.n_states), states_source_turbine)
        hh = fdata[FV.H][stsel]
        self.wmodel.contribute_to_wake_deltas(algo, mdata, fdata,
                pdata, states_source_turbine, wake_coos, wake_deltas)
        
        pdata[FC.POINTS] = pdata[FC.POINTS].copy() # making sure this is no ref
        
        for h in self.heights:

            fdata[FV.H][stsel] = hh + 2*(h - hh)

            nwcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, pdata, states_source_turbine)
            
            self.wmodel.contribute_to_wake_deltas(algo, mdata, fdata,
                    pdata, states_source_turbine, nwcoos, wake_deltas)
            
        fdata[FV.H][stsel] = hh

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        amb_results,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

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
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """
        self.wmodel.finalize_wake_deltas(algo, mdata, fdata,
            pdata, amb_results, wake_deltas)

class GroundMirror(WakeMirror):
    """
    A wake model wrapper that adds mirror turbines
    that model wake reflection from the ground

    :group: models.wake_models

    """
    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for WakeMirror
        kwargs: dict, optional
            Additional parameters for WakeMirror

        """
        super().__init__(*args, heights=[0], **kwargs)
