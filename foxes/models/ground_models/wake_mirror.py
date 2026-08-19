from __future__ import annotations

from foxes.core import GroundModel
import foxes.variables as FV
import foxes.constants as FC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.partial_wakes_model import PartialWakesModel
    from foxes.core.wake_model import WakeModel


class WakeMirror(GroundModel):
    """
    Wake reflection from ground and/or other horizontal planes.

    Attributes
    ----------
    heights
        The reflection heights


    """

    def __init__(self, heights: list[float]) -> None:
        """
        Constructor.

        Parameters
        ----------
        heights
            The reflection heights

        """
        super().__init__()
        self.heights = heights

    def contribute_to_farm_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
        pwake: PartialWakesModel,
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wake_deltas
            The wake deltas. Key: variable name,
            Values have shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel
            The wake model
        pwake
            The partial wakes model

        """
        # prepare:
        hh = fdata[FV.H][:, downwind_index].copy()

        # DEBUG CHECK:
        # import numpy as np
        # assert(np.all(fdata[FV.H]==fdata[FV.TXYH[..., 2]]))

        # contribution from main wake:
        wake_frame = getattr(algo, "wake_frame", None)
        if wake_frame is None:
            raise ValueError(
                f"WakeMirror '{self.name}': algorithm '{algo.name}' has no wake_frame"
            )
        wcoos = wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # contribution from mirrors:
        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref
        for h in self.heights:
            fdata[FV.TXYH][:, downwind_index, 2] = hh + 2 * (h - hh)

            pwake.contribute(
                algo, mdata, fdata, tdata, downwind_index, wake_deltas, wmodel
            )

        # reset heights:
        fdata[FV.TXYH][:, downwind_index, 2] = hh

    def contribute_to_point_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wake_deltas
            The wake deltas. Key: variable name,
            Values have shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel
            The wake model

        """
        # prepare:
        hh = fdata[FV.H][:, downwind_index].copy()

        # contribution from main wake:
        wake_frame = getattr(algo, "wake_frame", None)
        if wake_frame is None:
            raise ValueError(
                f"WakeMirror '{self.name}': algorithm '{algo.name}' has no wake_frame"
            )
        wcoos = wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # contribution from mirrors:
        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref
        for h in self.heights:
            fdata[FV.TXYH][:, downwind_index, 2] = hh + 2 * (h - hh)

            wcoos = wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
            wmodel.contribute(
                algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas
            )

        # reset heights:
        fdata[FV.TXYH][:, downwind_index, 2] = hh


class GroundMirror(WakeMirror):
    """
    Wake reflection from the ground.


    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        super().__init__(heights=[0.0])
