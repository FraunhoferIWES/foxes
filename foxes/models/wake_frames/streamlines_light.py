import numpy as np

from foxes.core import WakeFrame
from foxes.core import TData
from foxes.utils import wd2uv
from foxes.config import config

import foxes.variables as FV
import foxes.constants as FC


class StreamlinesLight(WakeFrame):
    """
    memory-light streamlines wake frame model

    Attributes
    ----------
    step: float
        The streamline step size in m
    chunksize_steps: int
        The number of steps per chunk for streamline
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation

    :group: models.wake_frames

    """

    def __init__(
            self, 
            step, 
            max_length_km=20, 
            chunksize_steps=100,
            cl_ipars={}, 
            **kwargs,
        ):
        """
        Constructor.

        Parameters
        ----------
        step: float
            The streamline step size in m
        max_length_km: float
            The maximal streamline length in km
        chunksize_steps: int
            The number of steps per chunk for streamline
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(max_length_km=max_length_km, **kwargs)
        self.step = step
        self.chunksize_steps = chunksize_steps
        self.cl_ipars = cl_ipars

        self.WPOINTS = self.var("wpoints")

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
        # prepare:
        n_states = fdata.n_states
        n_turbines = algo.n_turbines
        tdata = TData.from_points(points=fdata[FV.TXYH], mdata=mdata)

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = self.get_wake_coos(
            algo, mdata, fdata, tdata, downwind_index=None
        )[:, :, 0, 0].reshape(n_states, n_turbines, n_turbines)

        # derive turbine order:
        # TODO: Remove loop over states
        order = np.zeros((n_states, n_turbines), dtype=config.dtype_int)
        for si in range(n_states):
            order[si] = np.lexsort(keys=coosx[si])

        return order

    def _calc_streamlines(self, algo, mdata, fdata):
        """ Helper function for streamline calculation """
        # prepare:
        n_states = mdata.n_states
        n_turbines = algo.n_turbines
        n_steps = np.ceil((self.max_length_km * 1000) / self.step).astype(int) + 1
        states_ovars = algo.states.output_point_vars(algo)
        assert FV.WD in states_ovars and FV.WS in states_ovars, (
            f"Wake frame '{self.name}': Require '{FV.WD}' and '{FV.WS}' in states output, found {states_ovars}"
        )

        # compute streamline points, starting at rotor centres:
        wpoints = np.full((n_states, n_turbines, n_steps, 3), np.nan, dtype=config.dtype_double)
        wpoints[:, :, 0, :2] = fdata[FV.TXYH][:, :, :2]
        wpoints[:, :, :, 2] = fdata[FV.TXYH][:, :, None, 2]
        for i_step in range(1, n_steps):
            # get local wind vector directions:
            tdata = TData.from_points(points=wpoints[:, :, i_step - 1, :], mdata=mdata)
            wpres = algo.states.calculate(algo, mdata, fdata, tdata)
            nx = wd2uv(wpres[FV.WD][:, :, 0])
            del tdata, wpres

            # advance point:
            wpoints[:, :, i_step, :2] = wpoints[:, :, i_step - 1, :2] + nx * self.step

        # store points:
        mdata[self.WPOINTS] = wpoints[..., :2]
        mdata.dims[self.WPOINTS] = (FC.STATE, FC.TURBINE, self.var("step"), FC.XY)

    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index=None,
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
        downwind_index: int, optional
            The index of the wake causing turbine
            in the downwind order, or all if None

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
            if downwind_index is not None, else shape:
            (n_states, n_turbines * n_targets, n_tpoints, 3)

        """
        # compute streamlines if not already done:
        if self.WPOINTS not in mdata:
            self._calc_streamlines(algo, mdata, fdata)
        
        # prepare:
        n_states, n_targets, n_tpoints = tdata[FC.TARGETS].shape[:3]
        n_points = n_targets * n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_points, 3)
        n_trbns = 1 if downwind_index is not None else algo.n_turbines

        states_ovars = algo.states.output_point_vars(algo)
        assert FV.WD in states_ovars and FV.WS in states_ovars, (
            f"Wake frame '{self.name}': Require '{FV.WD}' and '{FV.WS}' in states output, found {states_ovars}"
        )

        # compute streamlines if not already done:
        if self.WPOINTS not in mdata:
            self._calc_streamlines(algo, mdata, fdata)
        wpoints = mdata[self.WPOINTS] if downwind_index is None else mdata[self.WPOINTS][:, downwind_index, None, :, :]
        n_steps = wpoints.shape[2]

        # compute coordinates:
        coos = np.full((n_states, n_trbns, n_points, 3), np.nan, dtype=config.dtype_double)
        heights = fdata[FV.TXYH] if downwind_index is None else fdata[FV.TXYH][:, downwind_index, None]
        coos[:, :, :, 2] = points[:, None, :, 2] - heights[:, :, None, 2]
        steps_0 = 0
        while steps_0 < n_steps:
            # extract steps chunk:
            steps_1 = min(steps_0 + self.chunksize_steps, n_steps)
            if n_steps - steps_1 < 2:   
                steps_1 = n_steps
            steps_s = slice(steps_0, steps_1)
            wpts = wpoints[:, :, steps_s, :2]

            # dims: (states, trbns, points, steps, xy)
            pdel = points[:, None, :, None, :2] - wpts[:, :, None, :, :2]
            nx = np.zeros_like(wpts)
            nx[:, :, 1:, :] = (wpts[:, :, 1:, :] - wpts[:, :, :-1, :]) / self.step
            nx[:, :, 0, :] = nx[:, :, 1, :]
            del wpts

            x = pdel[..., 0] * nx[:, :, None, :, 0] + pdel[..., 1] * nx[:, :, None, :, 1]
            selx = (x > -self.step) & (x < self.step)
            if np.any(selx):
                pdel = pdel[selx]
                w = np.where(selx)
                nx = nx[w[0], w[1], w[3], :]

                y = -pdel[:, 0] * nx[:, 1] + pdel[:, 1] * nx[:, 0]
                cy = coos[w[0], w[1], w[2], 1]
                sely = np.isnan(cy) | (np.abs(y) < np.abs(cy))
                if np.any(sely):
                    w = (w[0][sely], w[1][sely], w[2][sely], w[3][sely])
                    coos[w[0], w[1], w[2], 0] = x[selx][sely] + self.step * np.arange(steps_0, steps_1)[w[3]]
                    coos[w[0], w[1], w[2], 1] = y[sely]
                del w, y, cy, sely
            del pdel, x, selx, nx
            steps_0 = steps_1

        if downwind_index is None:
            coos = coos.reshape(n_states, algo.n_turbines * n_targets, n_tpoints, 3)
        else:
            coos = coos[:, 0, :, :].reshape(n_states, n_targets, n_tpoints, 3)

        return algo.wake_deflection.calc_deflection(
            algo, mdata, fdata, tdata, downwind_index, coos
        )

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
        wd = fdata[self.var_wd][:, downwind_index]
        n = np.append(wd2uv(wd, axis=-1), np.zeros_like(wd)[:, None], axis=-1)

        xyz = fdata[FV.TXYH][:, downwind_index]
        return xyz[:, None, :] + x[:, :, None] * n[:, None, :]
