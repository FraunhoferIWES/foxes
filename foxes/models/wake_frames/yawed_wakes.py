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

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        self.base_frame.initialize(algo, verbosity)
        super().initialize(algo, verbosity)

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

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        return self.base_frame.model_input_data(algo)

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
        # prepare:
        n_states = mdata.n_states
        n_points = points.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)

        # get unyawed results:
        xyz = self.base_frame.get_wake_coos(
            algo, mdata, fdata, states_source_turbine, points
        )
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]

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

        return xyz

    def finalize(self, algo, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level, 0 = silent

        """
        self.base_frame.finalize(algo, clear_mem, verbosity)
        super().finalize(algo, clear_mem, verbosity)
