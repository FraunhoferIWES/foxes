import numpy as np

from foxes.core import WakeK
from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC


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
    wake_k: foxes.core.WakeK
        Handler for the wake growth parameter k

    :group: models.wake_models.ti

    """

    def __init__(
        self,
        superposition,
        opening_angle=21.6,
        iec_type="2019",
        induction="Betz",
        **wake_k,
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The TI wake superposition.
        opening_angle: float, optional
            The wake opening angle. The wake growth parameter k is calculated
            based on the wake opening angle.
        iec_type: str
            Either '2005' or '2019'/'Frandsen'
        wake_k: dict, optional
            Parameters for the WakeK class

        """
        super().__init__(superpositions={FV.TI: superposition}, induction=induction)

        if opening_angle is not None:
            if "k" in wake_k or "ka" in wake_k or "kb" in wake_k:
                raise KeyError(
                    f"Can handle 'opening_angle' or ('k', 'ka', 'kb') parameters, not both"
                )
            wake_k["k"] = float(np.tan(np.deg2rad(opening_angle / 2.0)))

        self.iec_type = iec_type
        self.wake_k = WakeK(**wake_k)

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.superpositions[FV.TI]}, induction={iname}, "
        s += self.wake_k.repr() + ")"
        return s

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.wake_k]

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
        return {FV.TI: np.zeros_like(tdata[FC.TARGETS][..., 0])}

    def calc_wake_radius(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        ct,
    ):
        """
        Calculate the wake radius, depending on x only (not r).

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_targets)

        Returns
        -------
        wake_r: numpy.ndarray
            The wake radii, shape: (n_states, n_targets)

        """
        k = self.wake_k(
            FC.STATE_TARGET,
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=False,
            downwind_index=downwind_index,
        )
        return k * x

    def calc_centreline(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        x,
        wake_r,
        ct,
    ):
        """
        Calculate centre line results of wake deltas.

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
            The index in the downwind order
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        x: numpy.ndarray
            The x values, shape: (n_st_sel,)
        wake_r: numpy.ndarray
            The wake radii, shape: (n_st_sel,)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_st_sel,)

        Returns
        -------
        cl_del: dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_st_sel,)

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
            upcast=True,
        )[st_sel]

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )[st_sel]

        # calculate wind deficit:
        if self.iec_type == "2005":
            cl_deltas = np.sqrt(0.9) / (1.5 + 0.3 * x / D * np.sqrt(ws))
        elif self.iec_type == "2019" or self.iec_type == "Frandsen":
            cl_deltas = 1.0 / (1.5 + 0.8 * x / D / np.sqrt(ct))
        else:
            raise TypeError(
                f"Type of IEC {self.iec_type} not found. Select '2015' or '2019'/'Frandsen'."
            )

        return {FV.TI: cl_deltas}
