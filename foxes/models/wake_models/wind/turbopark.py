import numpy as np

from foxes.models.wake_models.gaussian import GaussianWakeModel
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
    k: float
        The wake growth parameter k.
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
        k=None,
        sbeta_factor=0.25,
        c1=1.5,
        c2=0.8,
        k_var=FV.K,
        induction="Madsen",
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind deficit superposition
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        sbeta_factor: float
            Factor multiplying sbeta
        c1: float
            Factor from Frandsen turbulence model
        c2: float
            Factor from Frandsen turbulence model
        k_var: str
            The variable name for k
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__(superpositions={FV.WS: superposition})

        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2
        self.k_var = k_var
        self.induction = induction

        setattr(self, k_var, k)

    def __repr__(self):
        k = getattr(self, self.k_var)
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.superpositions[FV.WS]}, induction={iname}"
        if k is None:
            s += f", k_var={self.k_var}"
        else:
            s += f", {self.k_var}={k}"
        s += ")"
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

    def calc_amplitude_sigma(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_st_sel,)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(st_sel):
            # apply selection:
            x = x[st_sel]
            ct = ct[st_sel]

            # get D:
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

            # get TI:
            ati = self.get_data(
                FV.AMB_TI,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )[st_sel]

            # get k:
            k = self.get_data(
                self.k_var,
                FC.STATE_TARGET,
                lookup="sw",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )[st_sel]

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
                + k
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
            st_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(st_sel)
            ampld = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, st_sel


class TurbOParkWakeIX(GaussianWakeModel):
    """
    The generalized TurbOPark wake model, integrating TI over the streamline.

    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Attributes
    ----------
    dx: float
        The step size of the integral
    k: float
        The wake growth parameter k.
    sbeta_factor: float
        Factor multiplying sbeta
    k_var: str
        The variable name for k
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
        k=None,
        sbeta_factor=0.25,
        k_var=FV.K,
        ti_var=FV.TI,
        self_wake=True,
        induction="Madsen",
        **ipars,
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind deficit superposition
        dx: float
            The step size of the integral
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        sbeta_factor: float
            Factor multiplying sbeta
        k_var: str
            The variable name for k
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
        self.sbeta_factor = sbeta_factor
        self.k_var = k_var
        self.ti_var = ti_var
        self.ipars = ipars
        self._tiwakes = None
        self.self_wake = self_wake
        self.induction = induction

        setattr(self, k_var, k)

    def __repr__(self):
        k = getattr(self, self.k_var)
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.superpositions[FV.WS]}, induction={iname}, dx={self.dx}"
        if k is None:
            s += f", k_var={self.k_var}"
        else:
            s += f", {self.k_var}={k}"
        s += f", ti_var={self.ti_var})"
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
        # find TI wake model:
        self._tiwakes = []
        for w in algo.wake_models.values():
            if w is not self:
                wdel = w.new_wake_deltas(algo, mdata, fdata, tdata)
                if self.ti_var in wdel:
                    self._tiwakes.append(w)
        if self.ti_var not in FV.amb2var and len(self._tiwakes) == 0:
            raise KeyError(
                f"Model '{self.name}': Missing wake model that computes wake delta for variable {self.ti_var}"
            )

        return super().new_wake_deltas(algo, mdata, fdata, tdata)

    def calc_amplitude_sigma(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_st_sel,)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-5) & (ct > 0.0)
        if np.any(st_sel):
            # apply selection:
            # x = x[st_sel]
            ct = ct[st_sel]

            # get D:
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

            # get k:
            k = self.get_data(
                self.k_var,
                FC.STATE_TARGET,
                lookup="sw",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )[st_sel]

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
                downwind_index,
                [self.ti_var],
                x,
                dx=self.dx,
                wake_models=self._tiwakes,
                self_wake=self.self_wake,
                **self.ipars,
            )[:, :, 0]

            # calculate sigma (eqn 1, plus epsilon from eqn 4 for x = 0)
            sigma = D * epsilon + k * ti_ix[st_sel]
            del x, epsilon

            # calculate amplitude, same as in Bastankhah model (eqn 7)
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

        # case no targets:
        else:
            st_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(st_sel)
            ampld = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, st_sel

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
