import numpy as np

from foxes.core import RotorModel
from foxes.tools import wd2uv, uv2wd
import foxes.variables as FV
import foxes.constants as FC

class CentreRotor(RotorModel):

    def n_rotor_points(self):
        return 1

    def design_points(self):
        return np.array([[0., 0., 0.]])
    
    def rotor_point_weights(self):
        return np.array([1.])

    def get_rotor_points(self, algo, fdata):
        return fdata[FV.TXYH][:, :, None, :]

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
            uv  = uvp[:, :, 0]

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

                if v == FV.REWS or v == FV.REWS2 or v == FV.REWS3:
                    rews = wsp[: ,:, 0]
                    if states_turbine is None:
                        out[v] = rews
                    else:
                        out[v][stsel] = rews[:, 0]
                    del rews

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in out:
                res = rpoint_results[v][:, :, 0]
                if states_turbine is None:
                    out[v] = res
                else:
                    out[v][stsel] = res[:, 0]
                del res
        
        return out