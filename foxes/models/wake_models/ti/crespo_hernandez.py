import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC


class CrespoHernandezTIWake(TopHatWakeModel):
    """
    The Crespo and Hernandez TI empirical correlation

    Source: https://doi.org/10.1016/0167-6105(95)00033-X

    For the wake diameter we use Eqns. (17), (15), (4), (5) from
            doi:10.1088/1742-6596/625/1/012039

    Attributes
    ----------
    k: float
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    a_near: float
        Model parameter
    a_far: float
        Model parameter
    e1: float
        Model parameter
    e2: float
        Model parameter
    e3: float
        Model parameter
    use_ambti: bool
        Flag for using ambient TI instead of local
        wake corrected TI
    sbeta_factor: float
        Factor multiplying sbeta
    near_wake_D: float
        The near wake distance in units of D,
        calculated from TI and ct if None
    k_var: str
        The variable name for k

    :group: models.wake_models.ti

    """

    def __init__(
        self,
        superposition,
        k=None,
        use_ambti=False,
        sbeta_factor=0.25,
        near_wake_D=None,
        ct_max=0.9999,
        a_near=0.362,
        a_far=0.73,
        e1=0.83,
        e2=-0.0325,
        e3=-0.32,
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
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        use_ambti: bool
            Flag for using ambient TI instead of local
            wake corrected TI
        sbeta_factor: float
            Factor multiplying sbeta
        near_wake_D: float, optional
            The near wake distance in units of D,
            calculated from TI and ct if not given here
        ct_max: float
            The maximal value for ct, values beyond will be limited
            to this number
        a_near: float
            Model parameter
        a_far: float
            Model parameter
        e1: float
            Model parameter
        e2: float
            Model parameter
        e3: float
            Model parameter
        k_var: str
            The variable name for k

        """
        super().__init__(superpositions={FV.TI: superposition}, ct_max=ct_max)

        self.a_near = a_near
        self.a_far = a_far
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.use_ambti = use_ambti
        self.sbeta_factor = sbeta_factor
        self.near_wake_D = near_wake_D
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

        # get D:
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
        sbeta = np.sqrt(0.5 * (1 + np.sqrt(1 - ct)) / np.sqrt(1 - ct))
        sblim = 1 / (np.sqrt(8) * self.sbeta_factor)
        sbeta[sbeta > sblim] = sblim
        radius = 2 * (k * x + self.sbeta_factor * sbeta * D)

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
        # prepare:
        n_states = fdata.n_states
        n_points = sp_sel.shape[1]
        n_targts = np.sum(sp_sel)
        st_sel = (np.arange(n_states), states_source_turbine)
        TI = FV.AMB_TI if self.use_ambti else FV.TI

        # read D from extra data:
        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][st_sel][:, None]
        D = D[sp_sel]

        # get ti:
        ti = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ti[:] = fdata[TI][st_sel][:, None]
        ti = ti[sp_sel]

        # prepare output:
        wake_deltas = np.zeros(n_targts, dtype=FC.DTYPE)

        # calc near wake length, if not given
        if self.near_wake_D is None:
            near_wake_D = (
                2**self.e1
                * self.a_near
                / (self.a_far * ti**self.e2)
                * (1 - np.sqrt(1 - ct)) ** (1 - self.e1)
            ) ** (1 / self.e3)
        else:
            near_wake_D = self.near_wake_D

        # calc near wake:
        sel = x < near_wake_D * D
        if np.any(sel):
            wake_deltas[sel] = self.a_near * (1.0 - np.sqrt(1.0 - ct[sel]))

        # calc far wake:
        if np.any(~sel):
            # calculate delta:
            #
            # Note the sign flip of the exponent ti[~sel]**(-0.0325)
            # compared to the original paper. This was found in
            # https://doi.org/10.1016/j.jweia.2018.04.010, Eq. (46)
            # Without this flip the near and far wake areas are not
            # smoothly connected.
            #
            wake_deltas[~sel] = (
                self.a_far
                * ((1.0 - np.sqrt(1.0 - ct[~sel])) / 2) ** self.e1
                * ti[~sel] ** self.e2
                * (x[~sel] / D[~sel]) ** self.e3
            )

        return {FV.TI: wake_deltas}
