import numpy as np

from foxes.core import RotorModel
from foxes.utils import wd2uv, uv2wd
import foxes.variables as FV
import foxes.constants as FC


class CentreRotor(RotorModel):
    """
    The centre rotor model.

    Evaluates states at a single point, located
    at the rotor centre.

    :group: models.rotor_models

    """

    def n_rotor_points(self):
        """
        The number of rotor points

        Returns
        -------
        n_rpoints: int
            The number of rotor points

        """
        return 1

    def design_points(self):
        """
        The rotor model design points.

        Design points are formulated in rotor plane
        (x,y,z)-coordinates in rotor frame, such that
        - (0,0,0) is the centre point,
        - (1,0,0) is the point radius * n_rotor_axis
        - (0,1,0) is the point radius * n_rotor_side
        - (0,0,1) is the point radius * n_rotor_up

        Returns
        -------
        dpoints: numpy.ndarray
            The design points, shape: (n_points, 3)

        """
        return np.array([[0.0, 0.0, 0.0]])

    def rotor_point_weights(self):
        """
        The weights of the rotor points

        Returns
        -------
        weights: numpy.ndarray
            The weights of the rotor points,
            add to one, shape: (n_rpoints,)

        """
        return np.array([1.0])

    def get_rotor_points(self, algo, mdata, fdata):
        """
        Calculates rotor points from design points.

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
        points: numpy.ndarray
            The rotor points, shape:
            (n_states, n_turbines, n_rpoints, 3)

        """
        return fdata[FV.TXYH][:, :, None, :]

    def eval_rpoint_results(
        self,
        algo,
        mdata,
        fdata,
        rpoint_results,
        weights,
        downwind_index=None,
        copy_to_ambient=False,
    ):
        """
        Evaluate rotor point results.

        This function modifies `fdata`, either
        for all turbines or one turbine per state,
        depending on parameter `states_turbine`. In
        the latter case, the turbine dimension of the
        `rpoint_results` is expected to have size one.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        rpoint_results: dict
            The results at rotor points. Keys: variable str.
            Values: numpy.ndarray, shape if `states_turbine`
            is None: (n_states, n_turbines, n_rpoints).
            Else: (n_states, 1, n_rpoints)
        weights: numpy.ndarray
            The rotor point weights, shape: (n_rpoints,)
        downwind_index: int, optional
            The index in the downwind order
        copy_to_ambient: bool
            If `True`, the fdata results are copied to ambient
            variables after calculation

        """
        if len(weights) > 1:
            return super().eval_rpoint_results(
                algo, mdata, fdata, rpoint_results, weights, downwind_index
            )

        n_states = mdata.n_states
        n_turbines = algo.n_turbines

        uvp = None
        uv = None
        if (
            FV.WS in self.calc_vars
            or FV.WD in self.calc_vars
            or FV.YAW in self.calc_vars
            or FV.REWS in self.calc_vars
            or FV.REWS2 in self.calc_vars
            or FV.REWS3 in self.calc_vars
        ):
            wd = rpoint_results[FV.WD]
            ws = rpoint_results[FV.WS]
            uvp = wd2uv(wd, ws, axis=-1)
            uv = uvp[:, :, 0]

        wd = None
        vdone = []
        for v in self.calc_vars:
            if v not in fdata:
                fdata[v] = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)

            if v == FV.WD or v == FV.YAW:
                if wd is None:
                    wd = uv2wd(uv, axis=-1)
                self._set_res(fdata, v, wd, downwind_index)
                vdone.append(v)

            elif v == FV.WS:
                self._set_res(fdata, v, ws[:, :, 0], downwind_index)
                del ws
                vdone.append(v)
        del uv, wd

        if (
            FV.REWS in self.calc_vars
            or FV.REWS2 in self.calc_vars
            or FV.REWS3 in self.calc_vars
        ):
            if downwind_index is None:
                yaw = fdata[FV.YAW]
            else:
                yaw = fdata[FV.YAW][:, downwind_index, None]
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum("stpd,std->stp", uvp, nax)

            for v in self.calc_vars:
                if v == FV.REWS or v == FV.REWS2 or v == FV.REWS3:
                    rews = wsp[:, :, 0]
                    self._set_res(fdata, v, rews, downwind_index)
                    del rews
                    vdone.append(v)

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in vdone:
                res = rpoint_results[v][:, :, 0]
                self._set_res(fdata, v, res, downwind_index)
                del res
            if copy_to_ambient and v in FV.var2amb:
                fdata[FV.var2amb[v]] = fdata[v].copy()
