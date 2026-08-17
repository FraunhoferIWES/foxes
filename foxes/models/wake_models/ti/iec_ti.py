from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import WakeK
from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import Model


class IECTIWake(TopHatWakeModel):
    """
    The TI wake model from IEC-64100-1-2005-8 (2005):

    Notes
    -----
    Reference:
    http://orbit.dtu.dk/files/3750291/2009_31.pdf
    v2: VolLuk: corrected implementation following: IEC-64100-1-2005-8
    (Appearently an error in the document by DTU)

    and the Frandsen wake TI model, from IEC-64100 (2019):
    http://orbit.dtu.dk/files/3750291/2009_31.pdf

    Attributes
    ----------
    wake_k
        Handler for the wake growth parameter k
    c0
        The c0 parameter for the wake decay
    c1
        The c1 parameter for the wake decay
    c2
        The c2 parameter for the wake decay

    :group: models.wake_models.ti

    """

    def __init__(
        self,
        superposition: str,
        opening_angle: float | None = 21.6,
        iec_type: str = "2019",
        induction: str = "Betz",
        c0: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
        **wake_k: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The TI wake superposition.
        opening_angle
            The wake opening angle. The wake growth parameter k is calculated
            based on the wake opening angle.
        iec_type
            Either '2005' or '2019'/'Frandsen'
        wake_k
            Parameters for the WakeK class
        induction
            The induction model to use. Default: 'Betz'
        c0
            The c0 parameter for the wake decay
        c1
            The c1 parameter for the wake decay
        c2
            The c2 parameter for the wake decay

        """
        super().__init__(
            other_superpositions={FV.TI: superposition}, induction=induction
        )
        self.iec_type = iec_type
        self.wake_k = None
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

        if opening_angle is None:
            self.wake_k = WakeK(**wake_k)
        else:
            if "k" in wake_k or "ka" in wake_k or "kb" in wake_k:
                raise KeyError(
                    "Can handle 'opening_angle' or ('k', 'ka', 'kb') parameters, not both"
                )
            self._k = float(np.tan(np.deg2rad(opening_angle / 2.0)))

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.other_superpositions[FV.TI]}, induction={iname}"
        if self.wake_k is not None:
            s += ", " + self.wake_k.repr()
        s += ")"
        return s

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return super().sub_models() + ([self.wake_k] if self.wake_k is not None else [])

    def new_wake_deltas(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

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

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return {FV.TI: np.zeros_like(tdata[FC.TARGETS][..., 0])}

    def calc_wake_radius(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        ct: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the wake radius, depending on x only (not r).

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
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        ct
            The ct values of the wake-causing turbines,
            shape: (n_states, n_targets)

        Returns
        -------
        wake_r
            The wake radii, shape: (n_states, n_targets)

        """
        if self.wake_k is None:
            return self._k * x
        else:
            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )

            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                upcast=False,
                downwind_index=downwind_index,
            )
            return D / 2 + k * x

    def calc_centreline(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        st_sel: np.ndarray,
        x: np.ndarray,
        wake_r: np.ndarray,
        ct: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Calculate centre line results of wake deltas.

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
            The index in the downwind order
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        x
            The x values, shape: (n_st_sel,)
        wake_r
            The wake radii, shape: (n_st_sel,)
        ct
            The ct values of the wake-causing turbines,
            shape: (n_st_sel,)

        Returns
        -------
        cl_del
            The centre line wake deltas. Key: variable name str,
            varlue

        """
        # read D from extra data:
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
            selection=st_sel,
        )

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
            selection=st_sel,
        )

        # calculate wind deficit:
        if self.iec_type == "2005":
            c0 = self.c0 if self.c0 is not None else np.sqrt(0.9)
            c1 = self.c1 if self.c1 is not None else 1.5
            c2 = self.c2 if self.c2 is not None else 0.3
            cl_deltas = c0 / (c1 + c2 * x / D * np.sqrt(ws))
        elif self.iec_type == "2019" or self.iec_type == "Frandsen":
            c0 = self.c0 if self.c0 is not None else 1.0
            c1 = self.c1 if self.c1 is not None else 1.5
            c2 = self.c2 if self.c2 is not None else 0.8
            cl_deltas = c0 / (c1 + c2 * x / D / np.sqrt(ct))
        else:
            raise TypeError(
                f"Type of IEC {self.iec_type} not found. Select '2005' or '2019'/'Frandsen'."
            )

        return {FV.TI: cl_deltas}
