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

    Parameters
    ----------
    k : float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data, by default None
    ct_max : float, optional
        The maximal value for ct, values beyond will be limited
        to this number, by default 0.9999
    alpha : float, optional
        model parameter used to determine onset of far wake region
    beta : float, optional
        model parameter used to determine onset of far wake region
    base_frame : foxes.core.WakeFrame
        The wake frame from which to start

    Attributes
    ----------
    model : PorteAgelModel
        The model for computing common data
    K : float
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    YAWM : float
        The yaw misalignment YAWM. If not given here
        it will be searched in the farm data.
    base_frame : foxes.core.WakeFrame
        The wake frame from which to start

    """

    def __init__(
        self,
        k=None,
        ct_max=0.9999,
        alpha=0.58,
        beta=0.07,
        base_frame=RotorWD(),
    ):
        super().__init__()

        self.base_frame = base_frame
        self.model = PorteAgelModel(ct_max, alpha, beta)

        setattr(self, FV.K, k)
        setattr(self, FV.YAWM, 0.0)

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
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().initialize(algo, verbosity)
        algo.update_idata(self.base_frame, idata=idata, verbosity=verbosity)

        return idata

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        order : numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        return self.base_frame.calc_order(algo, mdata, fdata)

    def _update_y(self, mdata, fdata, states_source_turbine, x, y):
        """
        Helper function for y deflection
        """
        # prepare:
        n_states = mdata.n_states
        n_points = x.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)

        # get gamma:
        gamma = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        gamma[:] = self.get_data(FV.YAWM, fdata, upcast="farm", data_prio=True)[st_sel][:, None]
        gamma *= np.pi / 180

        # get k:
        k = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        k[:] = self.get_data(FV.K, fdata, upcast="farm")[st_sel][:, None]

        # run model calculation:
        self.model.calc_data(mdata, fdata, states_source_turbine, x, gamma, k)

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

    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):
        """
        Calculate wake coordinates.

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
        points : numpy.ndarray
            The evaluation points, shape: (n_states, n_points, 3)

        Returns
        -------
        wake_coos : numpy.ndarray
            The wake coordinates, shape: (n_states, n_points, 3)

        """
        # get unyawed results:
        xyz = self.base_frame.get_wake_coos(
            algo, mdata, fdata, states_source_turbine, points
        )
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]

        # apply deflection:
        self._update_y(mdata, fdata, states_source_turbine, x, y)

        return xyz

    def get_centreline_points(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Gets the points along the centreline for given
        values of x.

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
            The wake frame x coordinates, shape: (n_states, n_points)
        
        Returns
        -------
        points : numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        points = self.base_frame.get_centreline_points(algo, mdata, fdata, 
                    states_source_turbine, x)

        nx = np.zeros_like(points)
        nx[:, 0] = points[:, 1] - points[:, 0]
        nx[:, -1] = points[:, -1] - points[:, -2]
        nx[:, 1:-1] = 0.5*(points[:, 1:-1] - points[:, :-2]) + 0.5*(points[:, 2:] - points[:, 1:-1])
        nx /= np.linalg.norm(nx, axis=-1)[:, :, None]

        nz = np.zeros_like(nx)
        nz[:, :, 2] = 1
        ny = np.cross(nz, nx, axis=-1)
        del nx, nz

        y = np.zeros_like(x)
        self._update_y(mdata, fdata, states_source_turbine, x, y)

        points += y[:, :, None] * ny
        
        return points

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        """
        if self.base_frame.initialized:
            self.base_frame.finalize(algo, verbosity)
        super().finalize(algo, verbosity)
