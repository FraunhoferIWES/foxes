
import numpy as np
from abc import abstractmethod

import foxes.variables as FV
import foxes.constants as FC
from foxes.core.farm_data_model import FarmDataModel
from foxes.core.turbine_model import TurbineModel
from foxes.core.farm_model import FarmModel
from foxes.core.data import Data
from foxes.tools import wd2uv, uv2wd


class RotorModel(FarmDataModel):

    def __init__(self, calc_vars):
        super().__init__()

        self.calc_vars = calc_vars

    def output_farm_vars(self, algo):
        return self.calc_vars + [FV.var2amb[v] for v in self.calc_vars if v in FV.var2amb]
    
    def initialize(self, algo):
        if not algo.states.initialized:
            algo.states.initialize(algo)
        super().initialize(algo)

    @abstractmethod
    def n_rotor_points(self):
        pass

    @abstractmethod
    def rotor_point_weights(self):
        pass

    @abstractmethod
    def design_points(self):
        pass

    def get_rotor_points(self, algo, mdata, fdata):

        n_states   = mdata.n_states
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
    
    def _set_res(self, fdata, v, res, stsel):
        if stsel is None:
            fdata[v] = res
        elif res.shape[1] == 1:
            fdata[v][stsel] = res[:, 0]
        else:
            fdata[v][stsel] = res[stsel]

    def eval_rpoint_results(
            self, 
            algo, 
            mdata,
            fdata, 
            rpoint_results, 
            weights,
            states_turbine=None,
            copy_to_ambient=False
        ):

        n_states   = mdata.n_states
        n_turbines = algo.n_turbines
        if states_turbine is not None:
            stsel = (np.arange(n_states), states_turbine)
        else:
            stsel = None

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

        wd    = None
        vdone = []
        for v in self.calc_vars:

            if v not in fdata:
                fdata[v] = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)

            if v == FV.WD or v == FV.YAW:
                if wd is None:
                    wd = uv2wd(uv, axis=-1)
                self._set_res(fdata, v, wd, stsel)
                vdone.append(v)

            elif v == FV.WS:
                ws = np.linalg.norm(uv, axis=-1)
                self._set_res(fdata, v, ws, stsel)
                del ws
                vdone.append(v)
        del uv, wd
        
        if FV.REWS in self.calc_vars \
            or FV.REWS2 in self.calc_vars \
            or FV.REWS3 in self.calc_vars:

            yaw = fdata[FV.YAW]
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum('stpd,std->stp', uvp, nax)

            for v in self.calc_vars:

                if v == FV.REWS:
                    rews = np.einsum('stp,p->st', wsp, weights)
                    self._set_res(fdata, v, rews, stsel)
                    del rews
                    vdone.append(v)

                elif v == FV.REWS2:
                    rews2 = np.sqrt(np.einsum('stp,p->st', wsp**2, weights))
                    self._set_res(fdata, v, rews2, stsel)
                    del rews2
                    vdone.append(v)

                elif v == FV.REWS3:
                    rews3 = (np.einsum('stp,p->st', wsp**3, weights))**(1./3.)
                    self._set_res(fdata, v, rews3, stsel)
                    del rews3
                    vdone.append(v)

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in vdone:
                res = np.einsum('stp,p->st', rpoint_results[v], weights)
                self._set_res(fdata, v, res, stsel)
            if copy_to_ambient and v in FV.var2amb:
                fdata[FV.var2amb[v]] = fdata[v].copy()

    def calculate(
            self, 
            algo, 
            mdata,
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
            rpoints = mdata.get(FV.RPOINTS, self.get_rotor_points(algo, mdata, fdata))
        if states_turbine is not None:
            n_states = mdata.n_states
            stsel    = (np.arange(n_states), states_turbine)
            rpoints  = rpoints[stsel][:, None]
        n_states, n_turbines, n_rpoints, __ = rpoints.shape
        n_points = n_turbines*n_rpoints

        if weights is None:
            weights = mdata.get(FV.RWEIGHTS, self.rotor_point_weights())
        
        if store_rpoints:
            mdata[FV.RPOINTS] = rpoints
        if store_rweights:
            mdata[FV.RWEIGHTS] = weights

        svars  = algo.states.output_point_vars(algo)
        points = rpoints.reshape(n_states, n_points, 3)
        pdata  = {FV.POINTS: points}
        pdims  = {FV.POINTS: (FV.STATE, FV.POINT, FV.XYH)}
        pdata.update({v: np.full((n_states, n_points), np.nan, dtype=FC.DTYPE) for v in svars})
        pdims.update({v: (FV.STATE, FV.POINT) for v in svars})
        pdata = Data(pdata, pdims, loop_dims=[FV.STATE, FV.POINT])
        del pdims, points

        algo.states.calculate(algo, mdata, fdata, pdata)

        rpoint_results = {}
        for v in svars:
            rpoint_results[v] = pdata[v].reshape(n_states, n_turbines, n_rpoints)

        if store_amb_res:
            mdata[FV.AMB_RPOINT_RESULTS] = rpoint_results

        self.eval_rpoint_results(algo, mdata, fdata, rpoint_results, 
                                    weights, states_turbine, copy_to_ambient=True)
        
        return {v: fdata[v] for v in self.output_farm_vars(algo)}
