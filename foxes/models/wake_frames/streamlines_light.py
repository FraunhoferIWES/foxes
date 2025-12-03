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
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation

    :group: models.wake_frames

    """

    def __init__(self, step, max_length_km=20, cl_ipars={}, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        step: float
            The streamline step size in m
        max_length_km: float
            The maximal streamline length in km
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(max_length_km=max_length_km, **kwargs)
        self.step = step
        self.cl_ipars = cl_ipars

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
        # prepare:
        n_states, n_targets, n_tpoints = tdata[FC.TARGETS].shape[:3]
        n_points = n_targets * n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_points, 3)
        trbns = [downwind_index] if downwind_index is not None else np.arange(algo.n_turbines)
        n_trbns = len(trbns)

        states_ovars = algo.states.output_point_vars(algo)
        assert FV.WD in states_ovars and FV.WS in states_ovars, (
            f"Wake frame '{self.name}': Require '{FV.WD}' and '{FV.WS}' in states output, found {states_ovars}"
        )

        # set starting points at rotor centre:
        wpoints = fdata[FV.TXYH][:, trbns].copy()
        coos = np.full((n_states, n_trbns, n_points, 3), np.nan, dtype=config.dtype_double)
        coos[:, :, :, 2] = points[:, None, :, 2] - wpoints[:, :, None, 2]
        wcntr = np.full((n_states, n_trbns, n_points), -1, dtype=config.dtype_int)
        wlength = 0.0
        counter = 0
        while True:
            # get local wind vector directions:
            htdata = TData.from_points(points=wpoints, mdata=mdata)
            wpres = algo.states.calculate(algo, mdata, fdata, htdata)
            wnx = wd2uv(wpres[FV.WD][:, :, 0])
            del htdata, wpres

            # project points:
            delp = points[:, None, :, :2] - wpoints[:, :, None, :2]
            x = np.einsum("stpd,std->stp", delp, wnx)
            selx = ((wcntr < 0) | (wcntr == counter - 1)) & (x >= -self.step) & (x < self.step)
            if np.any(selx):
                delp = delp[selx]
                wny = np.where(selx)
                wny = np.stack(
                    (-wnx[:, :, 1], wnx[:, :, 0]), axis=-1
                )[wny[0], wny[1]]
                y = np.einsum("sd,sd->s", delp, wny)
                cy = coos[selx, 1]
                sely = np.isnan(cy) | (np.abs(y) < np.abs(cy))
                if np.any(sely):
                    coos[selx, 1] = np.where(sely, y, cy)
                    coos[selx, 0] = np.where(sely, wlength + x[selx], coos[selx, 0])
                    wcntr[selx] = np.where(sely, counter, wcntr[selx])
                del y, cy, sely, wny
            del delp, x, selx

            # advance points:
            if (
                np.all((wcntr >= 0) & (wcntr < counter)) or
                wlength + self.step > self.max_length_km * 1000
            ):
                break
            else:
                wpoints[:, :, :2] += wnx * self.step
                wlength += self.step
                counter += 1

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
