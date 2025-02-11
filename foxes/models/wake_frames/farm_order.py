import numpy as np

from foxes.config import config
from foxes.core import WakeFrame

from .rotor_wd import RotorWD


class FarmOrder(WakeFrame):
    """
    Invokes turbine ordering as defined
    by the wind farm.

    Warning: This is for testing purposes only, and in general
    only gives correct calculation results when used
    in an iterative algorithm.

    Attributes
    ----------
    base_frame: foxes.core.WakeFrame
        The wake frame from which to start

    :group: models.wake_frames

    """

    def __init__(self, base_frame=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        base_frame: foxes.core.WakeFrame
            The wake frame from which to start
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.base_frame = base_frame

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
        if self.base_frame is None:
            self.base_frame = RotorWD()
        super().initialize(algo, verbosity, force)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.base_frame]

    def calc_order(self, algo, mdata, fdata):
        """
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        order = np.zeros((fdata.n_states, fdata.n_turbines), dtype=config.dtype_int)
        order[:] = np.arange(fdata.n_turbines)[None, :]

        return order

    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
        """
        Calculate wake coordinates of rotor points.

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
            in the downwind order

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        return self.base_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        return self.base_frame.get_centreline_points(
            algo, mdata, fdata, downwind_index, x
        )
