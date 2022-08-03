import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC


class JensenWake(TopHatWakeModel):
    """
    The Jensen wake model.

    Parameters
    ----------
    superpositions : dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book
    k : float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    ct_max : float
        The maximal value for ct, values beyond will be limited
        to this number

    Attributes
    ----------
    k : float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.

    """

    def __init__(self, superposition, k=None, ct_max=0.9999):
        super().__init__(superpositions={FV.WS: superposition}, ct_max=ct_max)

        setattr(self, FV.K, k)

    def __repr__(self):
        s = super().__repr__()
        s += f"(k={self.k}, sp={self.superp[FV.WS]})"
        return s

    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        """
        Initialize wake delta storage.

        They are added on the fly to the wake_deltas dict.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        n_points : int
            The number of wake evaluation points
        wake_deltas : dict
            The wake deltas storage, add wake deltas
            on the fly. Keys: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        n_states = mdata.n_states
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def calc_wake_radius(self, algo, mdata, fdata, states_source_turbine, x, ct):
        """
        Calculate the wake radius, depending on x only (not r).

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
            The x values, shape: (n_states, n_points)
        r : numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_points, n_r_per_x, 2)
        ct : numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_points)

        Returns
        -------
        wake_r : numpy.ndarray
            The wake radii, shape: (n_states, n_points)

        """
        n_states = mdata.n_states
        st_sel = (np.arange(n_states), states_source_turbine)

        R = fdata[FV.D][st_sel][:, None] / 2
        k = self.get_data(FV.K, fdata, st_sel)

        if isinstance(k, np.ndarray):
            k = k[:, None]

        return R + k * x

    def calc_centreline_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, sp_sel, x, wake_r, ct
    ):
        """
        Calculate centre line results of wake deltas.

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
        sp_sel : numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)
        x : numpy.ndarray
            The x values, shape: (n_sp_sel,)
        wake_r : numpy.ndarray
            The wake radii, shape: (n_sp_sel,)
        ct : numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_sp_sel,)

        Returns
        -------
        cl_del : dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_sp_sel,)

        """
        n_states = mdata.n_states
        n_points = sp_sel.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)

        R = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        R[:] = fdata[FV.D][st_sel][:, None] / 2
        R = R[sp_sel]

        return {FV.WS: -((R / wake_r) ** 2) * (1.0 - np.sqrt(1.0 - ct))}
