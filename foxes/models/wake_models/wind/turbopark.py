import numpy as np

from foxes.models.wake_models.gaussian import GaussianWakeModel
from foxes.utils import sqrt_reg
import foxes.variables as FV
import foxes.constants as FC


class TurbOParkWake(GaussianWakeModel):
    """
    The TurbOPark wake model

    Notes
    -----
    Reference:
    "Turbulence Optimized Park model with Gaussian wake profile"
    J G Pedersen, E Svensson, L Poulsen and N G Nygaard
    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Attributes
    ----------
    A: float
        The wake growth parameter A.
    sbeta_factor: float
        Factor multiplying sbeta
    c1: float
        Factor from Frandsen turbulence model
    c2: float
        Factor from Frandsen turbulence model
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.wind

    """

    def __init__(
        self,
        superposition,
        A,
        sbeta_factor=0.25,
        c1=1.5,
        c2=0.8,
        induction="Madsen",
    ):
        """
        Constructor.

        Parameters
        ----------
        superpositions: dict
            The superpositions. Key: variable name str,
            value: The wake superposition model name,
            will be looked up in model book
        A: float
            The wake growth parameter A.
        sbeta_factor: float
            Factor multiplying sbeta
        c1: float
            Factor from Frandsen turbulence model
        c2: float
            Factor from Frandsen turbulence model
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__(superpositions={FV.WS: superposition})

        self.A = A
        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2
        self.induction = induction

    def __repr__(self):
        s = super().__repr__()
        s += f"(A={self.A}, sp={self.superpositions[FV.WS]})"
        return s

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.induction]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

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
        wake_deltas[FV.WS] = np.zeros((n_states, pdata.n_points), dtype=FC.DTYPE)

    def calc_amplitude_sigma_spsel(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_sp_sel,)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):
            # apply selection:
            x = x[sp_sel]
            ct = ct[sp_sel]

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
            D = D[sp_sel]

            # get TI:
            ati = self.get_data(
                FV.AMB_TI,
                FC.STATE_POINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                pdata=pdata,
                upcast=True,
                states_source_turbine=states_source_turbine,
            )
            ati = ati[sp_sel]

            # calculate sigma:
            # beta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            a = self.induction.ct2a(ct)
            beta = (1 - a) / (1 - 2 * a)
            epsilon = self.sbeta_factor * np.sqrt(beta)
            del a, beta

            alpha = self.c1 * ati
            beta = self.c2 * ati / np.sqrt(ct)

            # calculate sigma (eqn 4)
            sigma = D * (
                epsilon
                + self.A
                * ati
                / beta
                * (
                    np.sqrt((alpha + beta * x / D) ** 2 + 1)
                    - np.sqrt(1 + alpha**2)
                    - np.log(
                        (np.sqrt((alpha + beta * x / D) ** 2 + 1) + 1)
                        * alpha
                        / ((np.sqrt(1 + alpha**2) + 1) * (alpha + beta * x / D))
                    )
                )
            )

            del (
                x,
                alpha,
                beta,
                epsilon,
            )

            # calculate amplitude, same as in Bastankhah model (eqn 7)
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

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

    Attributes
    ----------
    dx: float
        The step size of the integral
    A: float
        The wake growth parameter A.
    sbeta_factor: float
        Factor multiplying sbeta
    ti_var:  str
        The TI variable
    self_wake: bool
        Flag for considering only own wake in ti integral
    induction: foxes.core.AxialInductionModel or str
        The induction model
    ipars: dict
        Additional parameters for centreline integration

    :group: models.wake_models.wind

    """

    def __init__(
        self,
        superposition,
        dx,
        A,
        sbeta_factor=0.25,
        ti_var=FV.TI,
        self_wake=True,
        induction="Madsen",
        **ipars,
    ):
        """
        Constructor.

        Parameters
        ----------
        superpositions: dict
            The superpositions. Key: variable name str,
            value: The wake superposition model name,
            will be looked up in model book
        dx: float
            The step size of the integral
        A: float, optional
            The wake growth parameter A.
        sbeta_factor: float
            Factor multiplying sbeta
        ti_var:  str
            The TI variable
        self_wake: bool
            Flag for considering only own wake in ti integral
        induction: foxes.core.AxialInductionModel or str
            The induction model
        ipars: dict, optional
            Additional parameters for centreline integration

        """
        super().__init__(superpositions={FV.WS: superposition})

        self.dx = dx
        self.A = A
        self.sbeta_factor = sbeta_factor
        self.ti_var = ti_var
        self.ipars = ipars
        self._tiwakes = None
        self.self_wake = self_wake
        self.induction = induction

    def __repr__(self):
        s = super().__repr__()
        s += f"(ti={self.ti_var}, dx={self.dx}, A={self.A}, sp={self.superpositions[FV.WS]})"
        return s

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.induction]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

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
        wake_deltas[FV.WS] = np.zeros((n_states, pdata.n_points), dtype=FC.DTYPE)

        # find TI wake models:
        self._tiwakes = []
        for w in algo.wake_models:
            if w is not self:
                wdel = {}
                w.init_wake_deltas(algo, mdata, fdata, pdata, wdel)
                if self.ti_var in wdel:
                    self._tiwakes.append(w)
        if self.ti_var not in FV.amb2var and len(self._tiwakes) == 0:
            raise KeyError(
                f"Model '{self.name}': Missing wake model that computes wake delta for variable {self.ti_var}"
            )

    def calc_amplitude_sigma_spsel(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_sp_sel,)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # select targets:
        sp_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(sp_sel):
            # apply selection:
            # x = x[sp_sel]
            ct = ct[sp_sel]

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
            D = D[sp_sel]

            # calculate sigma:
            # beta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            a = self.induction.ct2a(ct)
            beta = (1 - a) / (1 - 2 * a)
            epsilon = self.sbeta_factor * np.sqrt(beta)
            del a, beta

            # get TI by integration along centre line:
            ti_ix = algo.wake_frame.calc_centreline_integral(
                algo,
                mdata,
                fdata,
                states_source_turbine,
                [self.ti_var],
                x,
                dx=self.dx,
                wake_models=self._tiwakes,
                self_wake=self.self_wake,
                **self.ipars,
            )[:, :, 0]

            # calculate sigma (eqn 1, plus epsilon from eqn 4 for x = 0)
            sigma = D * epsilon + self.A * ti_ix[sp_sel]

            del (
                x,
                sbeta,
                epsilon,
            )

            # calculate amplitude, same as in Bastankhah model (eqn 7)
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

        # case no targets:
        else:
            sp_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(sp_sel)
            ampld = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, sp_sel

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().finalize(algo, verbosity)
        self._tiwakes = None
