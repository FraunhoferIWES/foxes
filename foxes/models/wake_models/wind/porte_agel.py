import numpy as np

from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel
import foxes.variables as FV
import foxes.constants as FC


class PorteAgelModel:
    """
    Common calculations for the wake model and the wake
    frame, such that code repetitions can be avoided.

    Based on Bastankhah & Porte-Agel, 2016, https://doi.org/10.1017/jfm.2016.595

    Parameters
    ----------
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number, by default 0.9999
    alpha : float
        model parameter used to determine onset of far wake region
    beta : float
        model parameter used to determine onset of far wake region

    Attributes
    ----------
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number, by default 0.9999
    alpha : float
        model parameter used to determine onset of far wake region
    beta : float
        model parameter used to determine onset of far wake region

    """

    MDATA_KEY = "PorteAgelModel"
    PARS = "pars"
    CHECK = "check"
    SP_SEL = "sp_sel"
    X0 = "x0"

    NEAR = "near"
    R_PC = "r_pc"
    R_PC_S = "r_pc_s"
    AMPL_NEAR = "ampl_near"
    DELTA_NEAR = "delta_near"

    AMPL_FAR = "ampl_far"
    SIGMA_Y_FAR = "sigma_y_far"
    SIGMA_Z_FAR = "sigma_z_far"
    DELTA_FAR = "delta_far"

    def __init__(self, ct_max=0.9999, alpha=0.58, beta=0.07):
        self.ct_max = ct_max
        self.alpha = alpha
        self.beta = beta

    @property
    def pars(self):
        """
        Dictionary of the model parameters

        Returns
        -------
        dict :
            Dictionary of the model parameters

        """
        return dict(alpha=self.alpha, beta=self.beta, ct_max=self.ct_max)

    def calc_data(
        self,
        mdata,
        fdata,
        states_source_turbine,
        x,
        gamma,
        k,
    ):
        """
        Calculate common model data, store it in mdata.

        Parameters
        ----------
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x : numpy.ndarray
            The x values, shape: (n_states, n_points)
        gamma : numpy.ndarray
            The YAWM angles in radiants, shape: (n_states, n_points)
        k : numpy.ndarray
            The k parameter values, shape: (n_states, n_points)

        """
        # prepare:
        n_states = mdata.n_states
        n_points = x.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)

        # store parameters:
        out = {self.PARS: self.pars}
        out[self.CHECK] = (
            mdata[FV.STATE][0],
            states_source_turbine[0],
            x.shape,
        )

        # get D:
        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][st_sel][:, None]

        # get ct:
        ct = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = fdata[FV.CT][st_sel][:, None]
        ct[ct > self.ct_max] = self.ct_max

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):

            # get ws:
            ws = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            ws[:] = fdata[FV.REWS][st_sel][:, None]

            # get TI:
            ti = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            ti[:] = fdata[FV.TI][st_sel][:, None]

            # apply filter:
            x = x[sp_sel]
            D = D[sp_sel]
            ct = ct[sp_sel]
            ws = ws[sp_sel]
            ti = ti[sp_sel]
            k = k[sp_sel]
            gamma = gamma[sp_sel]

            # calc theta_c0, Eq. (6.12):
            cosg = np.cos(gamma)
            theta = 0.3 * gamma / cosg * (1 - np.sqrt(1 - ct * cosg))

            # calculate x0, Eq. (7.3):
            alpha = self.alpha
            beta = self.beta
            sqomct = np.sqrt(1 - ct)
            x0 = (
                D
                * (cosg * (1 + sqomct))
                / (np.sqrt(2) * (4 * alpha * ti + 2 * beta * (1 - sqomct)))
            )
            out[self.X0] = x0

            # calcuate sigma, Eq. (7.2):
            sigma_y0 = D * cosg / np.sqrt(8)
            simga_z0 = D / np.sqrt(8)
            sigma_y = k * (x - x0) + sigma_y0
            sigma_z = k * (x - x0) + simga_z0

            # calc near wake data:
            near = x < x0
            out[self.NEAR] = near
            if np.any(near):

                # apply filter:
                wsn = ws[near]
                ctn = ct[near]
                cosgn = cosg[near]

                # initial velocity deficits, Eq. (6.4):
                uR = 0.5 * ctn * cosgn / (1 - np.sqrt(1 - ctn * cosgn))

                # constant potential core value, Eq. (6.7):
                u0 = np.sqrt(1 - ctn)

                # compute potential core shape, for later, Eq. (6.13):
                d = x[near] / x0[near]
                r_pc_0 = 0.5 * D[near] * np.sqrt(uR / u0)  # radius at x=0
                r_pc = r_pc_0 - d * r_pc_0  # potential core radius

                # memorize near wake data:
                out[self.R_PC] = r_pc
                out[self.R_PC_S] = d * sigma_y0[near]
                out[self.DELTA_NEAR] = theta[near] * x[near]
                out[self.AMPL_NEAR] = u0 - 1

                # cleanup:
                del wsn, ctn, cosgn, uR, u0, d, r_pc_0, r_pc

            # calc far wake data:
            far = ~near
            if np.any(far):

                # apply filter:
                ws = ws[far]
                ct = ct[far]
                sigma_y = sigma_y[far]
                sigma_z = sigma_z[far]
                cosg = cosg[far]
                D = D[far]
                theta = theta[far]
                x0 = x0[far]
                k = k[far]
                sqomct = sqomct[far]

                # calculate delta, Eq. (7.4):
                sqct = np.sqrt(ct)
                sqsd = np.sqrt(8 * sigma_y * sigma_z / (cosg * D**2))
                delta = theta * x0 + (
                    D
                    * theta
                    / 14.7
                    * np.sqrt(cosg / (ct * k**2))
                    * (2.9 + 1.3 * sqomct - ct)
                    * np.log(
                        ((1.6 + sqct) * (1.6 * sqsd - sqct))
                        / ((1.6 - sqct) * (1.6 * sqsd + sqct))
                    )
                )

                # calculate amplitude, Eq. (7.1):
                ampl = np.sqrt(1 - ct * cosg * D**2 / (8 * sigma_y * sigma_z)) - 1

                # memorize far wake data:
                out[self.AMPL_FAR] = ampl
                out[self.DELTA_FAR] = delta
                out[self.SIGMA_Y_FAR] = sigma_y
                out[self.SIGMA_Z_FAR] = sigma_z

        # update mdata:
        out[self.SP_SEL] = sp_sel
        mdata[self.MDATA_KEY] = out

    def has_data(self, mdata, states_source_turbine, x):
        """
        Check if data exists

        Parameters
        ----------
        mdata : foxes.core.Data
            The model data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x : numpy.ndarray
            The x values, shape: (n_states, n_points)

        Returns
        -------
        check : bool
            True if data exists

        """
        check = (
            mdata[FV.STATE][0],
            states_source_turbine[0],
            x.shape,
        )
        return self.MDATA_KEY in mdata and mdata[self.MDATA_KEY][self.CHECK] == check

    def get_data(self, key, mdata):
        """
        Return data entry

        Parameters
        ----------
        key : str
            The data key
        mdata : foxes.core.Data
            The model data

        Returns
        -------
        data : numpy.ndarray
            The data

        """
        return mdata[self.MDATA_KEY][key]

    def clean(self, mdata):
        """
        Clean all data
        """
        del mdata[self.MDATA_KEY]


class PorteAgelWake(DistSlicedWakeModel):
    """
    The Bastankhah PorteAgel wake model

    Based on Bastankhah & Porte-Agel, 2016, https://doi.org/10.1017/jfm.2016.595


    Parameters
    ----------
    superposition : dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book
    k : float
        The wake growth parameter k. If not given here
        it will be searched in the farm data, by default None
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number, by default 0.9999
    alpha : float
        model parameter used to determine onset of far wake region
    beta : float
        model parameter used to determine onset of far wake region

    Attributes
    ----------
    model : PorteAgelModel
        The model for computing common data
    K : float
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    YAWM : float
        The yaw misalignment YAWM. If not given here
        it will be searched in the farm data.

    """

    def __init__(self, superposition, k=None, ct_max=0.9999, alpha=0.58, beta=0.07):
        super().__init__(superpositions={FV.WS: superposition})

        self.model = PorteAgelModel(ct_max, alpha, beta)

        setattr(self, FV.K, k)
        setattr(self, FV.YAWM, 0.0)

    def __repr__(self):
        s = super().__repr__()
        s += f"(k={self.k}, sp={self.superpositions[FV.WS]})"
        return s

    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        """
        Initialize wake delta storage.

        They are added on the fly to the wake_deltas dict.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        n_points : int
            The number of wake evaluation points
        wake_deltas : dict
            The wake deltas storage, add wake deltas
            on the fly. Keys: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        n_states = mdata.n_states
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def calc_wakes_spsel_x_yz(self, algo, mdata, fdata, states_source_turbine, x, yz):
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x : numpy.ndarray
            The x values, shape: (n_states, n_points)
        yz : numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_points, n_yz_per_x, 2)

        Returns
        -------
        wdeltas : dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_yz_per_x)
        sp_sel : numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        # prepare:
        n_states = mdata.n_states
        n_points = x.shape[1]
        n_y_per_z = yz.shape[2]
        st_sel = (np.arange(n_states), states_source_turbine)

        # calculate model data:
        if not self.model.has_data(mdata, states_source_turbine, x):

            # get gamma:
            gamma = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            gamma[:] = self.get_data(FV.YAWM, fdata, upcast="farm", data_prio=True)[st_sel][:, None]
            gamma *= np.pi / 180

            # get k:
            k = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            k[:] = self.get_data(FV.K, fdata, upcast="farm")[st_sel][:, None]

            # run calculation:
            self.model.calc_data(mdata, fdata, states_source_turbine, x, gamma, k)

        # select targets:
        sp_sel = self.model.get_data(PorteAgelModel.SP_SEL, mdata)
        n_sp_sel = np.sum(sp_sel)
        wdeltas = {FV.WS: np.zeros((n_sp_sel, n_y_per_z), dtype=FC.DTYPE)}
        if np.any(sp_sel):

            # apply filter:
            yz = yz[sp_sel]

            # collect data:
            near = self.model.get_data(PorteAgelModel.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):

                # collect data:
                ampl = self.model.get_data(PorteAgelModel.AMPL_NEAR, mdata)
                r_pc = self.model.get_data(PorteAgelModel.R_PC, mdata)
                s = self.model.get_data(PorteAgelModel.R_PC_S, mdata)

                # radial dependency:
                r = np.linalg.norm(yz[near], axis=-1)
                rfactor = np.ones_like(r)
                sel_oc = np.where(r > r_pc[:, None])
                r = r[sel_oc]
                r_pc = r_pc[sel_oc[0]]
                s = s[sel_oc[0]]
                rfactor[sel_oc] = np.exp(-0.5 * ((r - r_pc) / s) ** 2)

                # set deficit, Eq. (6.13):
                wdeltas[FV.WS][near] = ampl[:, None] * rfactor

            # far wake:
            if np.any(far):

                # apply filter:
                yz = yz[far]

                # collect data:
                ampl = self.model.get_data(PorteAgelModel.AMPL_FAR, mdata)[:, None]
                sigma_y = self.model.get_data(PorteAgelModel.SIGMA_Y_FAR, mdata)[
                    :, None
                ]
                sigma_z = self.model.get_data(PorteAgelModel.SIGMA_Z_FAR, mdata)[
                    :, None
                ]

                # set deficit, Eq. (7.1):
                y = yz[..., 0]
                z = yz[..., 1]
                wdeltas[FV.WS][far] = ampl * (
                    np.exp(-0.5 * (y / sigma_y) ** 2)
                    * np.exp(-0.5 * (z / sigma_z) ** 2)
                )

        return wdeltas, sp_sel
