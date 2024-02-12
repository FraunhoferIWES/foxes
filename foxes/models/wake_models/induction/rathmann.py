import numpy as np

from foxes.core import WakeModel
import foxes.variables as FV
import foxes.constants as FC


class Rathmann(WakeModel):
    """
    The Rathmann induction wake model

    The individual wake effects are superposed linearly,
    without invoking a wake superposition model.

    Notes
    -----
    Reference:
    Forsting, Alexander R. Meyer, et al. 
    "On the accuracy of predicting wind-farm blockage." 
    Renewable Energy (2023).
    https://www.sciencedirect.com/science/article/pii/S0960148123007620

    Attributes
    ----------
    pre_rotor_only: bool
        Calculate only the pre-rotor region
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(self, pre_rotor_only=False, induction="Madsen"):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor_only: bool
            Calculate only the pre-rotor region
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__()
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only

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
        n_points = pdata.n_points
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_coos,
        wake_deltas,
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

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
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """

        # get x, y and z
        x = wake_coos[:, :, 0]
        y = wake_coos[:, :, 1]
        z = wake_coos[:, :, 2]

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

        # get D
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

        x_R = x/(D/2)
        r_R = np.sqrt(y**2 + z**2)/(D/2)
        
        def mu(x_R):
            """ axial shape function at r=0 from vortex cylinder model (eqn 11) """
            return 1 + x_R/(np.sqrt(1+x_R**2))
        
        def G(x_R, r_R):
            """ radial shape function eqn 20 """
            sin_2_alpha = (2*x_R) / np.sqrt((x_R**2 +(r_R-1)**2)*(x_R**2+(r_R+1)**2)) # eqn 19
            sin_alpha = np.sqrt(0.5*(1-np.sqrt(1-sin_2_alpha**2))) # derived from cos(2a)**2 + sin(2a)**2 = 1
            sin_beta = 1/np.sqrt(x_R**2 + r_R**2 +1) # eqn 19
            return sin_alpha * sin_beta * (1+x_R**2)

        # ws delta in front of rotor
        sp_sel = (ct > 0) &  (x < 0) 
        if np.any(sp_sel):
            a = self.induction.ct2a(ct[sp_sel])
            blockage = ws[sp_sel] * a* mu(x_R[sp_sel]) * G(x_R[sp_sel], r_R[sp_sel]) # eqn 10
            wake_deltas[FV.WS][sp_sel] += -blockage

        # ws delta behind rotor
        if not self.pre_rotor_only:
            # mirror -blockage in rotor plane
            sp_sel = (ct > 0) & (x > 0) & (r_R > 1 )
            if np.any(sp_sel):
                a = self.induction.ct2a(ct[sp_sel])
                blockage = ws[sp_sel] * a * mu(-x_R[sp_sel]) * G(-x_R[sp_sel], r_R[sp_sel]) # eqn 10
                wake_deltas[FV.WS][sp_sel] += blockage

        return wake_deltas
