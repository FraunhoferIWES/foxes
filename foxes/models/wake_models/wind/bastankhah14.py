import numpy as np

from foxes.models.wake_models.gaussian import GaussianWakeModel
import foxes.variables as FV
import foxes.constants as FC


class Bastankhah2014(GaussianWakeModel):
    """
    The Bastankhah 2014 wake model

    Notes
    -----
    Reference:
    "A new analytical model for wind-turbine wakes"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1016/j.renene.2014.01.002

    Attributes
    ----------
    k: float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    sbeta_factor: float
        Factor multiplying sbeta
    k_var: str
        The variable name for k
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.wind

    """

    def __init__(
        self,
        superposition,
        k=None,
        sbeta_factor=0.2,
        k_var=FV.K,
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
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        sbeta_factor: float
            Factor multiplying sbeta
        k_var: str
            The variable name for k
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__(superpositions={FV.WS: superposition})

        self.sbeta_factor = sbeta_factor
        self.k_var = k_var
        self.induction = induction

        setattr(self, k_var, k)

    def __repr__(self):
        k = getattr(self, self.k_var)
        s = super().__repr__()
        s += f"({self.k_var}={k}, sp={self.superpositions[FV.WS]})"
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
        wake_deltas[FV.WS] = np.zeros((mdata.n_states, pdata.n_points), dtype=FC.DTYPE)

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
            k = k[sp_sel]

            # calculate sigma:
            # beta = 0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)
            a = self.induction.ct2a(ct)
            beta = (1 - a) / (1 - 2 * a)
            sigma = k * x + self.sbeta_factor * np.sqrt(beta) * D
            del beta, a

            # calculate amplitude:
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

        # case no targets:
        else:
            sp_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(sp_sel)
            ampld = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, sp_sel
