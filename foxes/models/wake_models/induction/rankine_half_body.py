import numpy as np

from foxes.core import WakeModel
import foxes.variables as FV
import foxes.constants as FC

class RHB(WakeModel):
    """
    The Rankine half body induction wake model

    Ref: B Gribben and G Hawkes - A potential flow model for wind turbine induction and wind farm blockage
    Techincal Paper, Frazer-Nash Consultancy, 2019

    """
    
    
    def __init__(self, superposition, ct_max=0.9999):
        super().__init__()

        self.ct_max = ct_max

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

    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_coos, wake_deltas):
    
        # get x, y and z
        x = wake_coos[:, :, 0]
        y = wake_coos[:, :, 1]
        z = wake_coos[:, :, 2]

        # get state and point data
        n_states = mdata.n_states
        n_points = x.shape[1]
        
        st_sel = (np.arange(n_states), states_source_turbine)

        # get ct:
        ct = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = self.get_data(FV.CT, fdata)[st_sel]
        ct[ct > self.ct_max] = self.ct_max

        # get ws:
        ws = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ws[:] = self.get_data(FV.REWS, fdata)[st_sel]

        # get D
        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = self.get_data(FV.D, fdata)[st_sel]

        # find a (page 6)
        a = 0.5 * (1-np.sqrt(1-ct))

        # find rotational area
        rotational_area = np.pi *(D/2)**2

        # find m (page 7)
        m = 2 * ws * a * rotational_area

        # get r and theta
        r = np.sqrt(y**2 + z**2)
        r_sph = np.sqrt(r**2 + x**2)
        theta = np.arctan2(r,x) # x has neg values, using abs, but always returns nans!
        #n_r_x = r.shape[2]

        # define rankine half body shape (page 3)
        RHB_shape = np.cos(theta) -(2/m) * np.pi * ws * (r_sph*np.sin(theta))**2

        # stagnation point condition
        xs  = -np.sqrt(m / (4 * np.pi * ws)) 

        # select targets
        sp_sel = (ct > 0) & ( ( RHB_shape < -1 ) | ( x < xs ) )

        # ws and wd delta storage
        ws_deltas = np.zeros((n_states, n_points, 4), dtype=FC.DTYPE)
        wd_deltas = np.zeros((n_states, n_points, 4), dtype=FC.DTYPE)

        if np.any(sp_sel):

            # apply selection
            x = x[sp_sel]
            y = y[sp_sel]
            z = z[sp_sel]
            ct = ct[sp_sel]
            D = D[sp_sel]
            ws = ws[sp_sel]
            m = m[sp_sel]

            # calc velocity components
            vel_factor =  m / (4*np.pi*(x**2 +y**2 + z**2)**1.5)
            u = vel_factor * x
            v = vel_factor * y
            w = vel_factor * z

            # calc wind direction in horizontal plane
            wd = (180 + np.rad2deg(np.arctan2(u, v)))%360

            ws_deltas[...,0][sp_sel] = x
            ws_deltas[...,1][sp_sel] = y
            ws_deltas[...,2][sp_sel] = z
            ws_deltas[...,3][sp_sel] = u

            wd_deltas[...,0][sp_sel] = x
            wd_deltas[...,1][sp_sel] = y
            wd_deltas[...,2][sp_sel] = z
            wd_deltas[...,3][sp_sel] = wd

        return {FV.WS: ws_deltas,
                 FV.WD: wd_deltas}



            








        print()
        

        