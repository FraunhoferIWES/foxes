import numpy as np

from foxes.core import TurbineInductionModel
from foxes.utils import uv2wd, wd2uv
import foxes.variables as FV
import foxes.constants as FC


class RankineHalfBody(TurbineInductionModel):
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

    def new_wake_deltas(self, algo, mdata, fdata, wpoints):
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        wpoints: numpy.ndarray
            The wake evaluation points,
            shape: (n_states, n_turbines, n_rpoints, 3)
        
        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled 
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return {
            FV.WS: np.zeros_like(wpoints[:, :, :, 0]),
            FV.WD: np.zeros_like(wpoints[:, :, :, 0]),
            "U": np.zeros_like(wpoints[:, :, :, 0]),
            "V": np.zeros_like(wpoints[:, :, :, 0])
        }
    
    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by 
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        # get x, y and z
        x = wake_coos[..., 0]
        y = wake_coos[..., 1]
        z = wake_coos[..., 2]

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )[:, :, True]

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )[:, :, None]

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )[:, :, None]

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
