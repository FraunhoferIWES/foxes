from foxes.core import GroundModel
import foxes.variables as FV
import foxes.constants as FC


class WakeMirror(GroundModel):
    """
    Wake reflection from ground and/or other horizontal planes.

    Attributes
    ----------
    heights: list of float
        The reflection heights

    :group: models.ground_models

    """

    def __init__(self, heights):
        """
        Constructor.

        Parameters
        ----------
        heights: list of float
            The reflection heights

        """
        super().__init__()
        self.heights = heights

    def contribute_to_farm_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
        pwake,
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
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        pwake: foxes.core.PartialWakesModel
            The partial wakes model

        """
        # prepare:
        hh = fdata[FV.H][:, downwind_index].copy()

        # contribution from main wake:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # contribution from mirrors:
        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref
        for h in self.heights:

            fdata[FV.H][:, downwind_index] = hh + 2 * (h - hh)

            pwake.contribute(
                algo, mdata, fdata, tdata, downwind_index, wake_deltas, wmodel
            )

        # reset heights:
        fdata[FV.H][:, downwind_index] = hh

    def contribute_to_point_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
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
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        # prepare:
        hh = fdata[FV.H][:, downwind_index].copy()

        # contribution from main wake:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # contribution from mirrors:
        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref
        for h in self.heights:

            fdata[FV.H][:, downwind_index] = hh + 2 * (h - hh)

            wcoos = algo.wake_frame.get_wake_coos(
                algo, mdata, fdata, tdata, downwind_index
            )
            wmodel.contribute(
                algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas
            )

        # reset heights:
        fdata[FV.H][:, downwind_index] = hh


class GroundMirror(WakeMirror):
    """
    Wake reflection from the ground.

    :group: models.ground_models

    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__(heights=[0.0])
