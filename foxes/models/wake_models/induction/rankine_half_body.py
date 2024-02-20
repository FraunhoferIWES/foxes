import numpy as np

from foxes.core import WakeModel
from foxes.utils import uv2wd, wd2uv
import foxes.variables as FV
import foxes.constants as FC


class RankineHalfBody(WakeModel):
    """
    The Rankine half body induction wake model

    The individual wake effects are superposed linearly,
    without invoking a wake superposition model.

    Notes
    -----
    Reference:
    B Gribben and G Hawkes
    "A potential flow model for wind turbine induction and wind farm blockage"
    Techincal Paper, Frazer-Nash Consultancy, 2019
    https://www.fnc.co.uk/media/o5eosxas/a-potential-flow-model-for-wind-turbine-induction-and-wind-farm-blockage.pdf

    Attributes
    ----------
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(self, induction="Madsen"):
        """
        Constructor.

        Parameters
        ----------
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__()
        self.induction = induction

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
        wake_deltas["U"] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        wake_deltas["V"] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        wake_deltas[FV.WD] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

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

        # calc m (page 7, skipping pi everywhere)
        m = 2 * ws * self.induction.ct2a(ct) * (D / 2) ** 2

        # get r and theta
        r = np.sqrt(y**2 + z**2)
        r_sph = np.sqrt(r**2 + x**2)
        theta = np.arctan2(r, x)

        # define rankine half body shape (page 3)
        RHB_shape = (
            np.cos(theta) - (2 / (m + 1e-15)) * ws * (r_sph * np.sin(theta)) ** 2
        )

        # stagnation point condition
        xs = -np.sqrt(m / (4 * ws))

        # set values out of body shape
        sp_sel = (ct > 0) & ((RHB_shape < -1) | (x < xs))
        if np.any(sp_sel):
            # apply selection
            xyz = wake_coos[sp_sel]

            # calc velocity components
            vel_factor = m[sp_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas["U"][sp_sel] += vel_factor * xyz[:, 0]
            wake_deltas["V"][sp_sel] += vel_factor * xyz[:, 1]

        # set values inside body shape
        sp_sel = (ct > 0) & (RHB_shape >= -1) & (x >= xs) & (x < 0)
        if np.any(sp_sel):
            # apply selection
            xyz = np.zeros_like(wake_coos[sp_sel])
            xyz[:, 0] = xs[sp_sel]

            # calc velocity components
            vel_factor = m[sp_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas["U"][sp_sel] += vel_factor * xyz[:, 0]

        return wake_deltas

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        amb_results,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

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
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """

        # calc ambient wind vector:
        ws0 = amb_results[FV.WS]
        nx = wd2uv(amb_results[FV.WD])
        wind_vec = nx * ws0[:, :, None]

        # wake deltas are in wake frame, rotate back to global frame:
        ny = np.stack((-nx[:, :, 1], nx[:, :, 0]), axis=2)
        delta_uv = wake_deltas["U"][:, :, None] * nx + wake_deltas["V"][:, :, None] * ny
        del ws0, nx, ny

        # add ambient result to wake deltas:
        wind_vec += delta_uv
        del delta_uv

        # deduce WS and WD deltas:
        new_wd = uv2wd(wind_vec)
        new_ws = np.linalg.norm(wind_vec, axis=-1)
        wake_deltas[FV.WS] += new_ws - amb_results[FV.WS]
        wake_deltas[FV.WD] += new_wd - amb_results[FV.WD]
