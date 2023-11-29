import numpy as np

from foxes.core import WakeModel
from foxes.utils import uv2wd, wd2uv
import foxes.variables as FV
import foxes.constants as FC


class RHB(WakeModel):
    """
    The Rankine half body induction wake model

    Ref: B Gribben and G Hawkes - A potential flow model for wind turbine induction and wind farm blockage
    Techincal Paper, Frazer-Nash Consultancy, 2019

    https://www.fnc.co.uk/media/o5eosxas/a-potential-flow-model-for-wind-turbine-induction-and-wind-farm-blockage.pdf

    :group: models.wake_models.induction

    """

    def __init__(self, superposition, ct_max=0.9999):
        super().__init__()
        self.superposition = superposition
        self.ct_max = ct_max

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.superp]

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.superp = algo.mbook.wake_superpositions[self.superposition]
        super().initialize(algo, verbosity)

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
        wake_deltas["W"] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
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
            lookup="f",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )
        ct[ct > self.ct_max] = self.ct_max

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_POINT,
            lookup="f",
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
            lookup="f",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # find a (page 6)
        a = 0.5 * (1 - np.sqrt(1 - ct))

        # find rotational area
        rotational_area = np.pi * (D / 2) ** 2

        # find m (page 7)
        m = 2 * ws * a * rotational_area

        # get r and theta
        r = np.sqrt(y**2 + z**2)
        r_sph = np.sqrt(r**2 + x**2)
        theta = np.arctan2(r, x)

        # define rankine half body shape (page 3)
        RHB_shape = np.cos(theta) - (2 / m) * np.pi * ws * (r_sph * np.sin(theta)) ** 2

        # stagnation point condition
        xs = -np.sqrt(m / (4 * np.pi * ws))

        # select targets
        sp_sel = (ct > 0) & ((RHB_shape < -1) | (x < xs))

        if np.any(sp_sel):
            # apply selection
            xyz = wake_coos[sp_sel]
            m = m[sp_sel]
            a = a[sp_sel]

            # calc velocity components
            vel_factor =  m / (4*np.pi*np.linalg.norm(xyz, axis=-1)**3)
            wake_deltas["U"][sp_sel] += (vel_factor[:, None] * xyz[:, :2])[:,0]
            wake_deltas["V"][sp_sel] += (vel_factor[:, None] * xyz[:, :2])[:,1]

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

        # calc ambient wind vector
        wind_vec = wd2uv(amb_results[FV.WD], amb_results[FV.WS])

        # add ambient result to wake deltas
        total_uv = wind_vec + np.stack((wake_deltas["U"], wake_deltas["V"]), axis=2)

        # deduce WS and WD deltas:
        # this does the job of calc_final_wake_delta, done here as not in parent class of RHB
        new_wd = uv2wd(total_uv)
        new_ws = np.linalg.norm(total_uv, axis=-1)
        wake_deltas[FV.WS] += new_ws - amb_results[FV.WS]
        wake_deltas[FV.WD] += new_wd - amb_results[FV.WD]
