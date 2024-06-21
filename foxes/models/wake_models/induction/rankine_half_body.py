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

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}(induction={iname})"

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
        return {
            FV.WS: np.zeros_like(tdata[FC.TARGETS][..., 0]),
            FV.WD: np.zeros_like(tdata[FC.TARGETS][..., 0]),
            "U": np.zeros_like(tdata[FC.TARGETS][..., 0]),
            "V": np.zeros_like(tdata[FC.TARGETS][..., 0]),
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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
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
        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # calc m (page 7, skipping pi everywhere)
        m = 2 * ws * self.induction.ct2a(ct) * (D / 2) ** 2

        # get r and theta
        x = np.round(wake_coos[..., 0], 12)
        r = np.linalg.norm(wake_coos[..., 1:], axis=-1)
        r_sph = np.sqrt(r**2 + x**2)
        theta = np.arctan2(r, x)

        # define rankine half body shape (page 3)
        RHB_shape = (
            np.cos(theta) - (2 / (m + 1e-15)) * ws * (r_sph * np.sin(theta)) ** 2
        )

        # stagnation point condition
        xs = -np.sqrt(m / (4 * ws + 1e-15))

        # set values out of body shape
        st_sel = (ct > 0) & ((RHB_shape < -1) | (x < xs))
        if np.any(st_sel):
            # apply selection
            xyz = wake_coos[st_sel]

            # calc velocity components
            vel_factor = m[st_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas["U"][st_sel] += vel_factor * xyz[:, 0]
            wake_deltas["V"][st_sel] += vel_factor * xyz[:, 1]

        # set values inside body shape
        st_sel = (ct > 0) & (RHB_shape >= -1) & (x >= xs) & (x <= 0)
        if np.any(st_sel):
            # apply selection
            xyz = np.zeros_like(wake_coos[st_sel])
            xyz[:, 0] = xs[st_sel]

            # calc velocity components
            vel_factor = m[st_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas["U"][st_sel] += vel_factor * xyz[:, 0]

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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        # calc ambient wind vector:
        ws0 = amb_results[FV.WS]
        nx = wd2uv(amb_results[FV.WD])
        wind_vec = nx * ws0[:, :, :, None]

        # wake deltas are in wake frame, rotate back to global frame:
        ny = np.stack((-nx[..., 1], nx[..., 0]), axis=-1)
        delta_uv = (
            wake_deltas["U"][:, :, :, None] * nx + wake_deltas["V"][:, :, :, None] * ny
        )
        del ws0, nx, ny

        # add ambient result to wake deltas:
        wind_vec += delta_uv
        del delta_uv

        # deduce WS and WD deltas:
        new_wd = uv2wd(wind_vec)
        new_ws = np.linalg.norm(wind_vec, axis=-1)
        wake_deltas[FV.WS] += new_ws - amb_results[FV.WS]
        wake_deltas[FV.WD] += new_wd - amb_results[FV.WD]
