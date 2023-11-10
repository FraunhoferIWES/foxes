import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC


class IECTIWake(TopHatWakeModel):
    """
    The TI wake model from IEC-64100-1-2005-8 (2005):

    source: http://orbit.dtu.dk/files/3750291/2009_31.pdf
    v2: VolLuk: corrected implementation following: IEC-64100-1-2005-8
    (Appearently an error in the document by DTU)

    and the Frandsen wake TI model, from IEC-64100 (2019):

    Source: http://orbit.dtu.dk/files/3750291/2009_31.pdf

    Attributes
    ----------
    opening_angle: float
        The wake opening angle. The wake growth parameter k is calculated
        based on the wake opening angle.
    k_var: str
        The variable name for k

    :group: models.wake_models.ti

    """

    def __init__(
        self,
        superposition,
        opening_angle=21.6,
        ct_max=0.9999,
        iec_type="2019",
        k_var=FV.K,
    ):
        """
        Constructor.

        Parameters
        ----------
        superpositions: dict
            The superpositions. Key: variable name str,
            value: The wake superposition model name,
            will be looked up in model book
        opening_angle: float
            The wake opening angle. The wake growth parameter k is calculated
            based on the wake opening angle.
        ct_max: float
            The maximal value for ct, values beyond will be limited
            to this number
        k_var: str
            The variable name for k

        """
        super().__init__(superpositions={FV.TI: superposition}, ct_max=ct_max)

        k = float(np.tan(np.deg2rad(opening_angle / 2.0)))

        self.iec_type = iec_type
        self.k_var = k_var
        setattr(self, k_var, k)

    def __repr__(self):
        k = getattr(self, self.k_var)
        s = super().__repr__()
        s += f"({self.k_var}={k}, sp={self.superpositions[FV.TI]})"
        return s

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
        n_states = mdata.n_states
        wake_deltas[FV.TI] = np.zeros((n_states, pdata.n_points), dtype=FC.DTYPE)

    def calc_wake_radius(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        ct,
    ):
        """
        Calculate the wake radius, depending on x only (not r).

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
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_points, n_r_per_x, 2)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_points)

        Returns
        -------
        wake_r: numpy.ndarray
            The wake radii, shape: (n_states, n_points)

        """

        # get k:
        k = self.get_data(
            self.k_var,
            FC.STATE_POINT,
            lookup="sw",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # calculate:
        radius = k * x

        return radius

    def calc_centreline_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        sp_sel,
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
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)
        x: numpy.ndarray
            The x values, shape: (n_sp_sel,)
        wake_r: numpy.ndarray
            The wake radii, shape: (n_sp_sel,)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_sp_sel,)

        Returns
        -------
        cl_del: dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_sp_sel,)

        """

        # read D from extra data:
        D = self.get_data(
            FV.D,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )
        D = D[sp_sel]

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )
        ws = ws[sp_sel]

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
