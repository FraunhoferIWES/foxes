import numpy as np

from foxes.models.wake_models.gaussian import GaussianWakeModel
import foxes.variables as FV
import foxes.constants as FC


class TurbOParkWake(GaussianWakeModel):
    """
    The TurbOPark wake model

    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Parameters
    ----------
    superpositions : dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book
    k : float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    sbeta_factor : float
        Factor multiplying sbeta
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number
    c1 : float
        Factor from Frandsen turbulence model
    c2 : float
        Factor from Frandsen turbulence model

    Attributes
    ----------
    k : float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    sbeta_factor : float
        Factor multiplying sbeta
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number
    c1 : float
        Factor from Frandsen turbulence model
    c2 : float
        Factor from Frandsen turbulence model

    """

    def __init__(
        self, superposition, k=None, sbeta_factor=0.25, ct_max=0.9999, c1=1.5, c2=0.8
    ):
        super().__init__(superpositions={FV.WS: superposition})

        self.ct_max = ct_max
        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2

        setattr(self, FV.K, k)

    def __repr__(self):
        s = super().__repr__()
        s += f"(k={self.k}, sp={self.superp[FV.WS]})"
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

    def calc_amplitude_sigma_spsel(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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

        Returns
        -------
        amsi : tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_sp_sel,)
        sp_sel : numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        # prepare:
        n_states = mdata.n_states
        n_points = x.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)

        # get ct:
        ct = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = self.get_data(FV.CT, fdata)[st_sel][:, None]
        ct[ct > self.ct_max] = self.ct_max

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):

            # apply selection:
            x = x[sp_sel]
            ct = ct[sp_sel]

            # get D:
            D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            D[:] = self.get_data(FV.D, fdata)[st_sel][:, None]
            D = D[sp_sel]

            # get k:
            k = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            k[:] = self.get_data(FV.K, fdata, upcast="farm")[st_sel][:, None]
            k = k[sp_sel]

            # get TI:
            AMB_TI = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            AMB_TI[:] = self.get_data(FV.AMB_TI, fdata)[st_sel][:, None]
            AMB_TI = AMB_TI[sp_sel]

            # calculate sigma:
            sbeta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            sblim = 1 / (np.sqrt(8) * self.sbeta_factor)
            sbeta[sbeta > sblim] = sblim
            epsilon = self.sbeta_factor * sbeta

            alpha = self.c1 * AMB_TI
            beta = self.c2 * AMB_TI / np.sqrt(ct)

            mult1 = k * AMB_TI / beta
            term1 = np.sqrt((alpha + beta * x / D) ** 2 + 1)
            term2 = np.sqrt(1 + alpha**2)
            term3 = (term1 + 1) * alpha
            term4 = (term2 + 1) * (alpha + beta * x / D)

            sigma = epsilon * D  # for x = 0

            sigma += sigma + D * mult1 * (term1 - term2 - np.log(term3 / term4))

            del (
                x,
                k,
                sbeta,
                sblim,
                mult1,
                term1,
                term2,
                term3,
                term4,
                alpha,
                beta,
                epsilon,
            )

            # calculate amplitude, same as in Bastankha model:
            if self.sbeta_factor < 0.25:
                radicant = 1.0 - ct / (8 * (sigma / D) ** 2)
                reals = radicant >= 0
                ampld = -np.ones_like(radicant)
                ampld[reals] = np.sqrt(radicant[reals]) - 1.0
            else:
                ampld = np.sqrt(1.0 - ct / (8 * (sigma / D) ** 2)) - 1.0

        # case no targets:
        else:
            sp_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(sp_sel)
            ampld = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, sp_sel
