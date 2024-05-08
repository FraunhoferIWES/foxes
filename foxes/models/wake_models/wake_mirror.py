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

    def new_wake_deltas(self, algo, mdata, fdata, tdata):
        """
        Creates new empty wake delta arrays.

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

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return self.wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        hh = fdata[FV.H][:, downwind_index].copy()
        self.wmodel.contribute(
            algo, mdata, fdata, tdata, downwind_index, wake_coos, wake_deltas
        )

        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref

        for h in self.heights:

            fdata[FV.H][:, downwind_index] = hh + 2 * (h - hh)

            nwcoos = algo.wake_frame.get_wake_coos(
                algo, mdata, fdata, tdata, downwind_index
            )

            self.wmodel.contribute(
                algo, mdata, fdata, tdata, downwind_index, nwcoos, wake_deltas
            )

        fdata[FV.H][:, downwind_index] = hh

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        self.wmodel.finalize_wake_deltas(
            algo,
            mdata,
            fdata,
            amb_results,
            wake_deltas,
        )


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
