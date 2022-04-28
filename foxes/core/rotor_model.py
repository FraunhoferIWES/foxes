
import numpy as np
from abc import abstractmethod

import foxes.variables as FV
import foxes.constants as FC
from foxes.core.farm_data_model import FarmDataModel
from foxes.core.point_data import PointData
from foxes.tools import wd2uv, uv2wd


class RotorModel(FarmDataModel):

    def __init__(self, calc_vars):
        super().__init__()
        self.calc_vars = calc_vars

    def output_farm_vars(self, algo):
        return self.calc_vars

    @abstractmethod
    def n_rotor_points(self):
        pass

    @abstractmethod
    def rotor_point_weights(self):
        pass

    @abstractmethod
    def design_points(self):
        pass

    def get_rotor_points(self, algo, fdata):

        n_states   = len(fdata[FV.STATE])
        n_points   = self.n_rotor_points()
        n_turbines = algo.n_turbines
        dpoints    = self.design_points()
        D          = fdata[FV.D]

        rax  = np.zeros((n_states, n_turbines, 3, 3), dtype=FC.DTYPE)
        n    = rax[:, :, 0, 0:2]
        m    = rax[:, :, 1, 0:2]
        n[:] = wd2uv(fdata[FV.YAW], axis=-1)
        m[:] = np.stack([-n[:, :, 1], n[:, :, 0]], axis=-1)
        rax[:, :, 2, 2] = 1

        points     = np.zeros((n_states, n_turbines, n_points, 3), dtype=FC.DTYPE)
        points[:]  = fdata[FV.TXYH][:, :, None, :]
        points[:] += 0.5 * D[:, :, None, None] * np.einsum('stad,pa->stpd', rax, dpoints)

        return points

    def eval_rpoint_results(
            self, 
            algo, 
            fdata, 
            rpoint_results, 
            weights,
            states_turbine=None
        ):

        n_states   = len(fdata[FV.STATE])
        n_turbines = algo.n_turbines
        if states_turbine is not None:
            stsel = (np.arange(n_states), states_turbine)

        uvp = None
        uv  = None
        if FV.WS in self.calc_vars \
            or FV.WD in self.calc_vars \
            or FV.YAW in self.calc_vars \
            or FV.REWS in self.calc_vars \
            or FV.REWS2 in self.calc_vars \
            or FV.REWS3 in self.calc_vars:

            wd  = rpoint_results[FV.WD]
            ws  = rpoint_results[FV.WS]
            uvp = wd2uv(wd, ws, axis=-1)
            uv  = np.einsum('stpd,p->std', uvp, weights)

        out = {}
        wd  = None
        for v in self.calc_vars:

            if states_turbine is not None:
                if v in fdata:
                    out[v] = fdata[v]
                else:
                    out[v] = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)

            if v == FV.WD or v == FV.YAW:
                if wd is None:
                    wd = uv2wd(uv, axis=-1)
                if states_turbine is None:
                    out[v] = wd
                else:
                    out[v][stsel] = wd[:, 0]

            elif v == FV.WS:
                ws = np.linalg.norm(uv, axis=-1)
                if states_turbine is None:
                    out[v] = ws
                else:
                    out[v][stsel] = ws[:, 0]
                del ws
        del uv, wd
        
        if FV.REWS in self.calc_vars \
            or FV.REWS2 in self.calc_vars \
            or FV.REWS3 in self.calc_vars:

            yaw = out.get(FV.YAW, fdata[FV.YAW])
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum('stpd,std->stp', uvp, nax)

            for v in self.calc_vars:

                if v == FV.REWS:
                    rews = np.einsum('stp,p->st', wsp, weights)
                    if states_turbine is None:
                        out[v] = rews
                    else:
                        out[v][stsel] = rews[:, 0]
                    del rews

                elif v == FV.REWS2:
                    rews2 = np.sqrt(np.einsum('stp,p->st', wsp**2, weights))
                    if states_turbine is None:
                        out[v] = rews2
                    else:
                        out[v][stsel] = rews2[:, 0]
                    del rews2

                elif v == FV.REWS3:
                    rews3 = (np.einsum('stp,p->st', wsp**3, weights))**(1./3.)
                    if states_turbine is None:
                        out[v] = rews3
                    else:
                        out[v][stsel] = rews3[:, 0]
                    del rews3

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in out:
                res = np.einsum('stp,p->st', rpoint_results[v], weights)
                if states_turbine is None:
                    out[v] = res
                else:
                    out[v][stsel] = res[:, 0]
                del res
        
        return out

    def calculate(
            self, 
            algo, 
            fdata, 
            rpoints=None, 
            weights=None, 
            store_rpoints=False, 
            store_rweights=False, 
            store_amb_res=False,
            states_turbine=None
        ):
        """ Calculate ambient results """

        if rpoints is None:
            rpoints = fdata.get(FV.RPOINTS, self.get_rotor_points(algo, fdata))
        if states_turbine is not None:
            n_states = len(fdata[FV.STATE])
            stsel    = (np.arange(n_states), states_turbine)
            rpoints = rpoints[stsel][:, None]
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        if weights is None:
            weights = fdata.get(FV.RWEIGHTS, self.rotor_point_weights())
        
        if store_rpoints:
            fdata[FV.RPOINTS] = rpoints
        if store_rweights:
            fdata[FV.RWEIGHTS] = weights

        points = rpoints.reshape(n_states, n_turbines*n_rpoints, 3)
        pdata  = PointData({FV.POINTS: points}, {FV.POINTS: (FV.STATE, FV.POINT)})
        point_results = algo.states.calculate(algo, fdata, pdata=pdata)
        del points, rpoints

        rpoint_results = {}
        for v, r in point_results.items():
            rpoint_results[v] = r.reshape(n_states, n_turbines, n_rpoints)
        del point_results

        if store_amb_res:
            fdata[FV.AMB_RPOINT_RESULTS] = rpoint_results

        return self.eval_rpoint_results(algo, fdata, rpoint_results, weights, states_turbine)
