import numpy as np

import foxes.constants as FC
from .rotor_wd import RotorWD
from foxes.core import WakeFrame

class FarmOrder(WakeFrame):
    """
    Invokes turbine ordering as defined
    by the wind farm.

    Warning: This is for testing purposes only, and in general
    only gives correct calculation results when used
    in an iterative algorithm.

    Parameters
    ----------
    base_frame : foxes.core.WakeFrame
        The wake frame from which to start

    Attributes
    ----------
    base_frame : foxes.core.WakeFrame
        The wake frame from which to start

    """

    def __init__(self, base_frame=RotorWD()):
        super().__init__()
        self.base_frame = base_frame

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
        order = np.zeros((fdata.n_states, fdata.n_turbines), dtype=FC.ITYPE)
        order[:] = np.arange(fdata.n_turbines)[None, :]

        return order

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
        return self.base_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)

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
        return self.base_frame.get_centreline_points(algo, mdata, fdata, states_source_turbine, x)
        
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
