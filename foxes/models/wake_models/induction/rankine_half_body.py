import numpy as np

from foxes.core import WakeModel
from foxes.utils import uv2wd
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
        self.superposition = superposition
        self.ct_max = ct_max

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
        self.superp = algo.mbook.wake_superpositions[self.superposition] 

        idata = super().initialize(algo, verbosity)
        algo.update_idata([self.superp], idata=idata, verbosity=verbosity)

        return idata
    
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
        wake_deltas[FV.WD] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

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
        theta = np.arctan2(r,x) 

        # define rankine half body shape (page 3)
        RHB_shape = np.cos(theta) -(2/m) * np.pi * ws * (r_sph*np.sin(theta))**2

        # stagnation point condition
        xs  = -np.sqrt(m / (4 * np.pi * ws)) 

        # select targets
        sp_sel = (ct > 0) & ( ( RHB_shape < -1 ) | ( x < xs ) )
        if np.any(sp_sel):

            # apply selection
            xyz = wake_coos[sp_sel] 
            m = m[sp_sel]

            # calc velocity components
            vel_factor =  m / (4*np.pi*np.linalg.norm(xyz, axis=-1)**3)
            uv = vel_factor[:, None] * xyz[:, :2]

            # adding wind direction linearly:
            wake_deltas[FV.WD][sp_sel] += uv2wd(uv, axis=-1)

            wake_deltas[FV.WS][sp_sel] -= np.linalg.norm(uv, axis=-1)

            # adding to wind speed deltas via superposition model
            """
            wake_deltas[FV.WS] = self.superp.calc_wakes_plus_wake(
                algo,
                mdata,
                fdata,
                states_source_turbine,
                sp_sel,
                FV.WS,
                wake_deltas[FV.WS],
                np.linalg.norm(uv, axis=-1),
            )
            """
            
        return wake_deltas

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        amb_results : dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas : dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """
        return
        wake_deltas[FV.WS] = self.superp.calc_final_wake_delta(
            algo, mdata, fdata, FV.WS, amb_results[FV.WS], wake_deltas[FV.WS]
        )

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
        algo.finalize_model(self.superp, verbosity)
        super().finalize(algo, verbosity)
