import numpy as np

from foxes.core import WakeFrame
from foxes.tools import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC

class Streamlines(WakeFrame):
    """
    Streamline following wakes

    Parameters
    ----------
    step : float
        The streamline step size in m
    n_delstor : int
        The streamline point storage increase
    
    Attributes
    ----------
    step : float
        The streamline step size in m
    n_delstor : int
        The streamline point storage increase

    """

    def __init__(self, step, n_delstor=100):
        super().__init__()
        self.step      = step
        self.n_delstor = n_delstor

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        self.DATA = self.var("DATA")
        self.CNTR = self.var("CNTR")
        super().initialize(algo, verbosity)

    def _calc_coos(self, algo, mdata, fdata, points, tcase=False):
        """
        Helper function, calculates streamline coordinates
        for given points.
        """

        # prepare:
        n_states   = mdata.n_states
        n_turbines = mdata.n_turbines
        n_points   = points.shape[1]
        n_spts     = mdata[self.CNTR]
        data       = mdata[self.DATA]
        spts       = data[..., :3]
        sn         = data[..., 3:6]
        slen       = data[..., 6]

        # find minimal distances to existing streamline points:
        # n_states, n_turbines, n_points, n_spts
        dists = np.linalg.norm(points[:, None, :, None] - spts[:, :, None, :n_spts], axis=-1)
        if tcase:
            for ti in range(n_turbines):
                dists[:, ti, ti] = 1e20
        inds  = np.argmin(dists, axis=3)
        dists = np.take_along_axis(dists, inds[:, :, :, None], axis=3)[..., 0]
        
        # add streamline points, as many as needed:
        done = inds < n_spts - 1
        while not np.all(done):

            # ensure storage size:
            if n_spts == data.shape[2]:
                data = np.append(data, np.full((n_states, n_turbines, self.n_delstor, 7), 
                                    np.nan, dtype=FC.DTYPE), axis=2)

                mdata[self.DATA] = data
                spts = data[..., :3]
                sn   = data[..., 3:6]
                slen = data[..., 6]

            # calculate next point:
            p0 = spts[:, :, n_spts - 1]
            n0 = sn[:, :, n_spts - 1]
            spts[:, :, n_spts] = p0 + self.step * n0
            slen[:, :, n_spts] = slen[:, :, n_spts - 1] + self.step
            newpts = spts[:, :, n_spts]
            del p0, n0

            # calculate next tangential vector:
            svars  = algo.states.output_point_vars(algo)
            pdata  = {FV.POINTS: newpts}
            pdims  = {FV.POINTS: (FV.STATE, FV.POINT, FV.XYH)}
            pdata.update({v: np.full((n_states, n_points), np.nan, dtype=FC.DTYPE) for v in svars})
            pdims.update({v: (FV.STATE, FV.POINT) for v in svars})
            pdata = Data(pdata, pdims, loop_dims=[FV.STATE, FV.POINT])
            data[:, :, n_spts, 3:5] = wd2uv(algo.states.calculate(algo, mdata, fdata, pdata)[FV.WD])
            data[:, :, n_spts, 5]   = 0. 
            del pdims, svars, pdata

            # evaluate distance:
            d = np.linalg.norm(points[:, None] - newpts[:, :, None], axis=-1)
            if tcase:
                for ti in range(n_turbines):
                    d[:, ti, ti] = 1e20
            sel = d < dists
            if np.any(sel):
                dists[sel] = d[sel]
                inds[sel]  = n_spts
                
            # rotation:
            mdata[self.CNTR] += 1
            n_spts = mdata[self.CNTR]
            done   = inds < n_spts - 1
        
        # shrink to size:
        mdata[self.DATA] = data[:, :, :n_spts]
        data = mdata[self.DATA]
        spts = data[..., :3]
        sn   = data[..., 3:6]
        slen = data[..., 6]

        # select:
        print("HERE",mdata[self.DATA].shape, inds.shape)


        print("INDS",inds)
        print("SLEN", slen)



        quit()



    def calc_order(self, algo, mdata, fdata):
        """"
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        
        Returns
        -------
        order : numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """

        print("\nHERE STRLNS CORDER")

        # prepare:
        n_states   = mdata.n_states
        n_turbines = mdata.n_turbines 

        # initialize storage:
        if self.DATA not in mdata:

            # x, y, z, u, v, w, len
            mdata[self.DATA] = np.full((n_states, n_turbines, self.n_delstor, 7), np.nan, dtype=FC.DTYPE)
            mdata[self.CNTR] = 1
 
            # set streamline start point data (rotor centre):
            mdata[self.DATA][:, :, 0, :3]  = fdata[FV.TXYH]
            mdata[self.DATA][:, :, 0, 3:5] = wd2uv(fdata[FV.AMB_WD])
            mdata[self.DATA][:, :, 0, 5]   = 0.
            mdata[self.DATA][:, :, 0, 6]   = 0.
        
        coos = self._calc_coos(algo, mdata, fdata, fdata[FV.TXYH], tcase=True)
        print("STRLNS CORDER: COOS")
        print(coos)
        quit()


    def _next_point(self, algo, mdata, fdata, coos, done, points, stsel, case_id):
        """
        Helper function, calculates and evaluates next streamline point
        """

        # prepare:
        n_states = mdata.n_states
        data     = mdata[self.DATA][case_id]
        inds     = mdata[self.CNTR][case_id]
        dist     = mdata[self.DIST][case_id]
        
        # ensure storage:
        if np.any(~np.isnan(data[:, -1])):
            data = np.append(data, np.full((n_states, self.n_delstor, 6), 
                                np.nan, dtype=FC.DTYPE), axis=1)
            mdata[self.DATA][case_id] = data

        # look at subset of states of interest:
        ssel  = ~np.all(done, axis=1)
        sinds = inds[ssel]
        sdata = data[ssel]
        sdist = dist[ssel]
        spts  = points[ssel]

        # calculate next point:
        ldata = np.take_along_axis(sdata, sinds[:, None, None], axis=1)[:, 0]
        p0    = ldata[..., :3]
        n     = ldata[..., 3:]
        p     = p0 + self.step * n
        d     = np.linalg.norm(spts - p[:, None], axis=2)
        del ldata, p0, n

        # this point is better:
        bsel = d < sdist
        if np.any(bsel):

            svars  = algo.states.output_point_vars(algo)
            points = rpoints.reshape(n_states, n_points, 3)
            pdata  = {FV.POINTS: points}
            pdims  = {FV.POINTS: (FV.STATE, FV.POINT, FV.XYH)}
            pdata.update({v: np.full((n_states, n_points), np.nan, dtype=FC.DTYPE) for v in svars})
            pdims.update({v: (FV.STATE, FV.POINT) for v in svars})
            pdata = Data(pdata, pdims, loop_dims=[FV.STATE, FV.POINT])
            del pdims, points
            
            algo.states.calculate(algo, mdata, fdata, pdata)


        print("\nHERE STRLNS CCOOS")
        print(sdata.shape, sinds.shape)
        print(mdata[self.DIST][case_id])
        print(ldata.shape)
        quit()

        
    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        points : numpy.ndarray
            The evaluation points, shape: (n_states, n_points, 3)
        
        Returns
        -------
        wake_coos : numpy.ndarray
            The wake coordinates, shape: (n_states, n_points, 3)

        """

        # prepare:
        n_states = mdata.n_states
        n_points = points.shape[1]
        stsel    = (np.arange(n_states), states_source_turbine)

        # init data:
        if self.DATA not in mdata:
            mdata[self.DATA] = {}
            mdata[self.CNTR] = {}
            mdata[self.DIST] = {}
        case_id = states_source_turbine[0]
        data = mdata[self.DATA]
        if case_id not in data:

            # x, y, z, nx, ny, nz
            data[case_id] = np.full((n_states, self.n_delstor, 6), np.nan, dtype=FC.DTYPE) 

            data[case_id][:, 0, :3]  = fdata[FV.TXYH][stsel]
            data[case_id][:, 0, 3:5] = wd2uv(fdata[FV.AMB_WD][stsel])
            data[case_id][:, 0, 5]   = 0.

            mdata[self.CNTR][case_id] = np.zeros(n_states, dtype=FC.ITYPE)
            mdata[self.DIST][case_id] = np.linalg.norm(points - data[case_id][:, 0, None, :3], axis=2) 

        # calculate coordinates, stored on the fly in mdata:
        coos = np.full((n_states, n_points, 3), np.nan, dtype=FC.DTYPE)
        done = np.zeros((n_states, n_points), dtype=bool)
        while not np.all(done):
            self._next_point(algo, mdata, fdata, coos, done, points, stsel, case_id)

        return coos
