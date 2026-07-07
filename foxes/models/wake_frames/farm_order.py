from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.core import WakeFrame

from .rotor_wd import RotorWD

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model


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

    def __init__(self, base_frame: WakeFrame | None = None, **kwargs: Any) -> None:
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

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if self.base_frame is None:
            self.base_frame = RotorWD()
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [] if self.base_frame is None else [self.base_frame]

    def calc_order(self, algo: Algorithm, mdata: MData, fdata: FData) -> np.ndarray:
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
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
    ) -> np.ndarray:
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
        bframe = self.base_frame
        assert bframe is not None, "Base wake frame not initialized"
        return bframe.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)

    def get_centreline_points(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        downwind_index: int,
        x: np.ndarray,
    ) -> np.ndarray:
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
        bframe = self.base_frame
        assert bframe is not None, "Base wake frame not initialized"
        return bframe.get_centreline_points(algo, mdata, fdata, downwind_index, x)
