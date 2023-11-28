import numpy as np

from foxes.core import WakeFrame
from foxes.models.wake_models.wind.porte_agel import PorteAgelModel
import foxes.variables as FV
import foxes.constants as FC
from .rotor_wd import RotorWD


class YawedWakes(WakeFrame):
    """
    Bend the wakes for yawed turbines.

    Based on Bastankhah & Porte-Agel, 2016, https://doi.org/10.1017/jfm.2016.595

    Attributes
    ----------
    model: PorteAgelModel
        The model for computing common data
    K: float
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    YAWM: float
        The yaw misalignment YAWM. If not given here
        it will be searched in the farm data.
    base_frame: foxes.core.WakeFrame
        The wake frame from which to start
    k_var: str
        The variable name for k

    :group: models.wake_frames

    """

    def __init__(
        self,
        k=None,
        ct_max=0.9999,
        alpha=0.58,
        beta=0.07,
        base_frame=RotorWD(),
        k_var=FV.K,
    ):
        """
        Constructor.

        Parameters
        ----------
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data, by default None
        ct_max: float, optional
            The maximal value for ct, values beyond will be limited
            to this number, by default 0.9999
        alpha: float, optional
            model parameter used to determine onset of far wake region
        beta: float, optional
            model parameter used to determine onset of far wake region
        base_frame: foxes.core.WakeFrame
            The wake frame from which to start
        k_var: str
            The variable name for k

        """
        super().__init__()

        self.base_frame = base_frame
        self.model = PorteAgelModel(ct_max, alpha, beta)
        self.k_var = k_var

        setattr(self, k_var, k)
        setattr(self, FV.YAWM, 0.0)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.base_frame]

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        return self.base_frame.calc_order(algo, mdata, fdata)

    def _update_y(self, algo, mdata, fdata, pdata, states_source_turbine, x, y):
        """
        Helper function for y deflection
        """

        # get gamma:
        gamma = self.get_data(
            FV.YAWM,
            FC.STATE_POINT,
            lookup="fs",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )
        gamma *= np.pi / 180

        # get k:
        k = self.get_data(
            self.k_var,
            FC.STATE_POINT,
            lookup="sf",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # run model calculation:
        self.model.calc_data(
            algo, mdata, fdata, pdata, states_source_turbine, x, gamma, k
        )

        # select targets:
        sp_sel = self.model.get_data(PorteAgelModel.SP_SEL, mdata)
        if np.any(sp_sel):
            # prepare:
            n_sp_sel = np.sum(sp_sel)
            ydef = np.zeros((n_sp_sel,), dtype=FC.DTYPE)

            # collect data:
            near = self.model.get_data(PorteAgelModel.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):
                # collect data:
                delta = self.model.get_data(PorteAgelModel.DELTA_NEAR, mdata)

                # set deflection:
                ydef[near] = delta

            # far wake:
            if np.any(far):
                # collect data:
                delta = self.model.get_data(PorteAgelModel.DELTA_FAR, mdata)

                # set deflection:
                ydef[far] = delta

            # apply deflection:
            y[sp_sel] -= ydef

    def get_wake_coos(self, algo, mdata, fdata, pdata, states_source_turbine):
        """
        Calculate wake coordinates.

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

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)

        """

        # get unyawed results:
        xyz = self.base_frame.get_wake_coos(
            algo, mdata, fdata, pdata, states_source_turbine
        )
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]

        # apply deflection:
        self._update_y(algo, mdata, fdata, pdata, states_source_turbine, x, y)

        return xyz

    def get_centreline_points(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        points = self.base_frame.get_centreline_points(
            algo, mdata, fdata, states_source_turbine, x
        )

        nx = np.zeros_like(points)
        nx[:, 0] = points[:, 1] - points[:, 0]
        nx[:, -1] = points[:, -1] - points[:, -2]
        nx[:, 1:-1] = 0.5 * (points[:, 1:-1] - points[:, :-2]) + 0.5 * (
            points[:, 2:] - points[:, 1:-1]
        )
        nx /= np.linalg.norm(nx, axis=-1)[:, :, None]

        nz = np.zeros_like(nx)
        nz[:, :, 2] = 1
        ny = np.cross(nz, nx, axis=-1)
        del nx, nz

        y = np.zeros_like(x)
        self._update_y(algo, mdata, fdata, None, states_source_turbine, x, y)

        points += y[:, :, None] * ny

        return points
