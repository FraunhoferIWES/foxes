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
    A : float
        The wake growth parameter A.
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
    A : float
        The wake growth parameter A.
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
        self, superposition, A, sbeta_factor=0.25, ct_max=0.9999, c1=1.5, c2=0.8
    ):
        super().__init__(superpositions={FV.WS: superposition})

        self.A = A
        self.ct_max = ct_max
        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        s = super().__repr__()
        s += f"(A={self.A}, sp={self.superpositions[FV.WS]})"
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
        ct[:] = fdata[FV.CT][st_sel][:, None]
        ct[ct > self.ct_max] = self.ct_max

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):

            # apply selection:
            x = x[sp_sel]
            ct = ct[sp_sel]

            # get D:
            D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            D[:] = fdata[FV.D][st_sel][:, None]
            D = D[sp_sel]

            # get TI:
            ati = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            ati[:] = fdata[FV.AMB_TI][st_sel][:, None]
            ati = ati[sp_sel]

            # calculate sigma:
            sbeta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            #sblim = 1 / (np.sqrt(8) * self.sbeta_factor)
            #sbeta[sbeta > sblim] = sblim
            epsilon = self.sbeta_factor * sbeta

            alpha = self.c1 * ati
            beta = self.c2 * ati / np.sqrt(ct)

            # calculate sigma (eqn 4)
            sigma = D * (
                epsilon
                + self.A*ati/beta
                * (
                    np.sqrt((alpha + beta*x/D)**2 + 1)
                    - np.sqrt(1 + alpha**2)
                    - np.log(
                        (np.sqrt((alpha + beta*x/D)**2 + 1) + 1)*alpha
                        / ((np.sqrt(1 + alpha**2) + 1) * (alpha + beta * x / D))
                    )
                )
            )

            del (
                x,
                sbeta,
                #sblim,
                alpha,
                beta,
                epsilon,
            )

            # calculate amplitude, same as in Bastankha model (eqn 7)
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

class TurbOParkWakeIX(GaussianWakeModel):
    """
    The generalized TurbOPark wake model, integrating TI over the streamline.

    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Parameters
    ----------
    superpositions : dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book
    dx : float
        The step size of the integral
    A : float, optional
        The wake growth parameter A.
    sbeta_factor : float
        Factor multiplying sbeta
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number
    ti_var :  str
        The TI variable
    ipars : dict, optional
        Additional parameters for centreline integration


    Attributes
    ----------
    dx : float
        The step size of the integral
    A : float
        The wake growth parameter A.
    sbeta_factor : float
        Factor multiplying sbeta
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number
    ti_var :  str
        The TI variable
    ipars : dict
        Additional parameters for centreline integration

    """

    def __init__(
        self, 
        superposition, 
        dx,
        A, 
        sbeta_factor=0.25, 
        ct_max=0.9999, 
        ti_var=FV.TI,
        **ipars,
    ):
        super().__init__(superpositions={FV.WS: superposition})

        self.dx = dx
        self.A = A
        self.ct_max = ct_max
        self.sbeta_factor = sbeta_factor
        self.ti_var = ti_var
        self.ipars = ipars

    def __repr__(self):
        s = super().__repr__()
        s += f"(ti={self.ti_var}, dx={self.dx}, A={self.A}, sp={self.superpositions[FV.WS]})"
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
        ct[:] = fdata[FV.CT][st_sel][:, None]
        ct[ct > self.ct_max] = self.ct_max

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):

            # apply selection:
            #x = x[sp_sel]
            ct = ct[sp_sel]

            # get D:
            D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            D[:] = fdata[FV.D][st_sel][:, None]
            D = D[sp_sel]

            # calculate sigma:
            sbeta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            #sblim = 1 / (np.sqrt(8) * self.sbeta_factor)
            #sbeta[sbeta > sblim] = sblim
            epsilon = self.sbeta_factor * sbeta

            # get TI by integratiion along centre line:
            ti_ix = algo.wake_frame.calc_centreline_integral(algo, mdata, fdata, 
                                states_source_turbine, [self.ti_var], x, 
                                dx=self.dx, **self.ipars)[:, :, 0]

            # calculate sigma (eqn 1, plus epsilon from eqn 4 for x = 0)
            sigma = D * epsilon + self.A * ti_ix[sp_sel]

            del (
                x,
                sbeta,
                #sblim,
                epsilon,
            )

            # calculate amplitude, same as in Bastankha model (eqn 7)
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
